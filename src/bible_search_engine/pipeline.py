# Author: Ronen Huang

from bible_search_engine.components.get_bible_data import get_bible_data
from bible_search_engine.components.preprocess import NLPTokenizer
from bible_search_engine.components.indexing import BibleChapterIndex, create_bible_index
from bible_search_engine.components.ranker import (TFIDFRanker, BM25Ranker, DirichletLMRanker, BiEncoderRanker,
                                                   CrossEncoderRanker, ColbertRanker)
from bible_search_engine.components.l2r import L2RRanker, L2RFeatureExtractor
from tqdm import tqdm, trange
import orjson
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
from importlib.resources import files
import os
from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
from typing import Literal


class BibleSearchEngine:
    def __init__(self, version: Literal["csv", "esv", "kjv", "msg", "nas", "niv", "nkjv", "nlt", "nrs"]="niv",
                 train_mode: bool=False, keep: set[str]=set()) -> None:
        '''
        Initializes the Bible search engine.

        For training, should be run from src directory.
        '''
        # Preprocess Bible data (if needed).
        bi_encoder_model = SentenceTransformer('msmarco-distilbert-dot-v5', device='cpu')

        if train_mode:
            old_testament_path = 'bible_search_engine/bible_data/old_testament_' + version + '.jsonl'
            new_testament_path = 'bible_search_engine/bible_data/new_testament_' + version + '.jsonl'

            if "get_bible_data" not in keep:
                get_bible_data(version)

            # NLP tokenizer.
            nlp_tokenizer = NLPTokenizer()

            # Bible chapter index.
            bible_chapter_index_path = 'bible_search_engine/bible_index_' + version

            if "indexing" not in keep:
                bible_chapter_index = create_bible_index(old_testament_path, new_testament_path, nlp_tokenizer)
                bible_chapter_index.save(bible_chapter_index_path)
            else:
                bible_chapter_index = BibleChapterIndex()
                bible_chapter_index.load(bible_chapter_index_path)

            # Traditional rankers.
            tf_idf_ranker = TFIDFRanker(bible_chapter_index, nlp_tokenizer)
            bm25_ranker = BM25Ranker(bible_chapter_index, nlp_tokenizer)
            dirichlet_lm_ranker = DirichletLMRanker(bible_chapter_index, nlp_tokenizer)

            # Cross-encoder ranker.
            #cross_encoder_ranker = self.get_cross_encoder(open(old_testament_path, 'rb'),
            #                                              open(new_testament_path, 'rb'))

            # Colbert ranker.
            if "colbert" not in keep:
                chapter_ids, chapter_texts = self.get_chapter_encodings(open(old_testament_path, 'rb'),
                                                                        open(new_testament_path, 'rb'), True)
                colbert_model, colbert_client = self.get_colbert(chapter_ids, chapter_texts)
            colbert_ranker = ColbertRanker("colbert-ir/colbertv2.0", "bible_chapters")

            # Learning to Rank features.
            l2r_feature_extractor = L2RFeatureExtractor(bible_chapter_index, nlp_tokenizer, tf_idf_ranker, bm25_ranker,
                                                        dirichlet_lm_ranker, colbert_ranker)

            encoded_chapters_path = 'bible_search_engine/bible_data/encoded_chapters_' + version + '.npy'
            chapter_ids_path = 'bible_search_engine/bible_data/chapterids_' + version + '.json'

            if "biencoder" not in keep:
                chapter_ids, chapter_texts = self.get_chapter_encodings(open(old_testament_path, 'rb'),
                                                                        open(new_testament_path, 'rb'), False)
                encoded_chapters = bi_encoder_model.encode(chapter_texts, show_progress_bar=True)
                np.save(encoded_chapters_path, encoded_chapters)
                if not os.path.isfile(chapter_ids_path):
                    with open(chapter_ids_path, 'xb') as chapterids_file:
                        chapterids_file.write(orjson.dumps(chapter_ids))
            else:
                encoded_chapters = np.load(encoded_chapters_path)
                with open(chapter_ids_path, "rb") as chapter_ids_file:
                    chapter_ids = orjson.loads(chapter_ids_file.readline())
            bi_encoder_ranker = BiEncoderRanker('msmarco-distilbert-dot-v5', encoded_chapters, chapter_ids)

            # Learning to Rank ranker.
            self.l2r_ranker = L2RRanker(bible_chapter_index, nlp_tokenizer, bi_encoder_ranker, l2r_feature_extractor)
            self.l2r_ranker.train('bible_search_engine/bible_queries_relevances/train_queries_relevances.csv',
                                  'bible_search_engine/bible_queries_relevances/test_queries_relevances.csv')
            joblib.dump(self.l2r_ranker.lightgbm_ranker, "bible_search_engine/initial_ranker_" + version + ".pkl", True)

            # Bible chapter titles and verses.
            self.get_chapter_verses(open(old_testament_path, 'rb'), open(new_testament_path, 'rb'))
        else:
            # NLP tokenizer.
            nlp_tokenizer = NLPTokenizer()

            # Bible chapter index.
            bible_chapter_index_path = files("bible_search_engine.bible_index_" + version)
            bible_chapter_index = BibleChapterIndex()
            bible_chapter_index.load(bible_chapter_index_path)

            # Traditional rankers.
            tf_idf_ranker = TFIDFRanker(bible_chapter_index, nlp_tokenizer)
            bm25_ranker = BM25Ranker(bible_chapter_index, nlp_tokenizer)
            dirichlet_lm_ranker = DirichletLMRanker(bible_chapter_index, nlp_tokenizer)

            # Cross-encoder ranker.
            old_testament_file = 'old_testament_' + version + '.jsonl'
            new_testament_file = 'new_testament_' + version + '.jsonl'
            old_testament_path = files("bible_search_engine.bible_data") / old_testament_file
            new_testament_path = files("bible_search_engine.bible_data") / new_testament_file

            #cross_encoder_ranker = self.get_cross_encoder(old_testament_path.open('rb'), new_testament_path.open('rb'))

            # Colbert ranker.
            colbert_ranker = ColbertRanker("colbert-ir/colbertv2.0", "bible_chapters")

            # Learning to Rank features.
            l2r_feature_extractor = L2RFeatureExtractor(bible_chapter_index, nlp_tokenizer,
                                                        tf_idf_ranker, bm25_ranker,
                                                        dirichlet_lm_ranker, colbert_ranker)

            # Bi-encoder ranker.
            encoded_chapters_file = 'encoded_chapters_' + version + '.npy'
            encoded_chapters_path = files("bible_search_engine.bible_data") / encoded_chapters_file
            chapter_ids_file = 'chapterids_' + version + '.json'
            chapter_ids_path = files("bible_search_engine.bible_data") / chapter_ids_file

            encoded_chapters = np.load(encoded_chapters_path)
            with chapter_ids_path.open('rb') as chapter_ids_file:
                chapter_ids = orjson.loads(chapter_ids_file.readline())
            bi_encoder_ranker = BiEncoderRanker('msmarco-distilbert-dot-v5', encoded_chapters, chapter_ids)

            # Learning to Rank ranker.
            self.l2r_ranker = L2RRanker(bible_chapter_index, nlp_tokenizer, bi_encoder_ranker, l2r_feature_extractor)
            ranker_file = "initial_ranker_" + version + ".pkl"
            self.l2r_ranker.lightgbm_ranker = joblib.load(files("bible_search_engine") / ranker_file)

            # Bible chapter titles and verses.
            self.get_chapter_verses(old_testament_path.open('rb'), new_testament_path.open('rb'))

    def get_cross_encoder(self, old_testament_file, new_testament_file) -> CrossEncoderRanker:
        verses = {}
        for bible_chapter_line in tqdm(old_testament_file):
            bible_chapter = orjson.loads(bible_chapter_line)
            verses[bible_chapter['chapterid']] = [verse for verse in bible_chapter['verses'].values()]
        for bible_chapter_line in tqdm(new_testament_file):
            bible_chapter = orjson.loads(bible_chapter_line)
            verses[bible_chapter['chapterid']] = [verse for verse in bible_chapter['verses'].values()]
        old_testament_file.close()
        new_testament_file.close()
        return CrossEncoderRanker('cross-encoder/msmarco-MiniLM-L6-en-de-v1', verses)

    def get_colbert(self, chapter_ids: list[int], chapter_texts: list[str]) -> None:
        colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
        chapter_embeddings = list(colbert_model.embed(chapter_texts))
        colbert_client = QdrantClient(
            url="https://aabdb742-af72-430a-b07d-ac0fb2cdf7d8.us-east4-0.gcp.cloud.qdrant.io:6333",
            api_key="---"
        )
        colbert_client.create_collection(
            collection_name="bible_chapters",
            vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE,
                                               multivector_config=models.MultiVectorConfig(
                                                   comparator=models.MultiVectorComparator.MAX_SIM)))
        for i in trange(0, len(chapter_ids), 10):
            start = i
            end = min(len(chapter_ids), i + 10)
            colbert_client.upsert(
                collection_name="bible_chapters",
                points=[models.PointStruct(id=chapter_id, vector=chapter_embedding) for chapter_id, chapter_embedding
                        in zip(chapter_ids[start:end], chapter_embeddings[start:end])])
        return colbert_model, colbert_client

    def get_chapter_encodings(self, old_testament_file, new_testament_file, complete: bool) ->\
            tuple[list[int], list[str]]:
        chapter_ids = []
        chapter_texts = []
        for bible_chapter_line in tqdm(old_testament_file):
            bible_chapter = orjson.loads(bible_chapter_line)
            if not complete:
                verse_chunk = ""
                for i, verse in enumerate(bible_chapter['verses'].values()):
                    verse_chunk += " " + verse
                    if (i + 1) % 5 == 0:
                        chapter_ids.append(bible_chapter['chapterid'])
                        chapter_texts.append(verse_chunk.strip())
                if len(verse_chunk) > 0:
                    chapter_ids.append(bible_chapter['chapterid'])
                    chapter_texts.append(verse_chunk.strip())
            else:
                chapter_ids.append(bible_chapter["chapterid"])
                chapter_texts.append(" ".join(bible_chapter["verses"].values()))
        for bible_chapter_line in tqdm(new_testament_file):
            bible_chapter = orjson.loads(bible_chapter_line)
            if not complete:
                verse_chunk = ""
                for i, verse in enumerate(bible_chapter['verses'].values()):
                    verse_chunk += " " + verse
                    if (i + 1) % 5 == 0:
                        chapter_ids.append(bible_chapter['chapterid'])
                        chapter_texts.append(verse_chunk.strip())
                if len(verse_chunk) > 0:
                    chapter_ids.append(bible_chapter['chapterid'])
                    chapter_texts.append(verse_chunk.strip())
            else:
                chapter_ids.append(bible_chapter["chapterid"])
                chapter_texts.append(" ".join(bible_chapter["verses"].values()))
        old_testament_file.close()
        new_testament_file.close()
        return chapter_ids, chapter_texts

    def get_chapter_verses(self, old_testament_file, new_testament_file) -> None:
        self.chapter_titles = {}
        self.chapter_verses = {}
        for bible_chapter_line in tqdm(old_testament_file):
            bible_chapter = orjson.loads(bible_chapter_line)
            chapterid = bible_chapter['chapterid']
            self.chapter_titles[chapterid] = bible_chapter['chapter']
            self.chapter_verses[chapterid] = bible_chapter['verses']
        for bible_chapter_line in tqdm(new_testament_file):
            bible_chapter = orjson.loads(bible_chapter_line)
            chapterid = bible_chapter['chapterid']
            self.chapter_titles[chapterid] = bible_chapter['chapter']
            self.chapter_verses[chapterid] = bible_chapter['verses']
        old_testament_file.close()
        new_testament_file.close()

    def search(self, query: str) -> list[dict]:
        '''
        query: Query of interest.

        Search for the query with the Bible search engine.
        '''
        query_results = self.l2r_ranker.query(query)
        search_results = [{'chapterid': chapterid, 'score': score, 'chapter': self.chapter_titles[chapterid],
                           'verses': self.chapter_verses[chapterid]}
                          for chapterid, score in query_results]
        return search_results


def create_bible_search_engine(version: str="niv", train_mode: bool=False,
                               keep: set[str]=set()) -> BibleSearchEngine:
    '''
    Creates the Bible search engine.

    Returns the created Bible search engine.
    '''
    return BibleSearchEngine(version, train_mode, keep)
