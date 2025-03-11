# Author: Ronen H

from bible_search_engine.preprocess_bible_data import get_bible_data
from bible_search_engine.document_preprocessor import NLPTokenizer
from bible_search_engine.indexing import BibleChapterIndex, create_bible_index
from bible_search_engine.ranker import TFIDFRanker, BM25Ranker, DirichletLMRanker, BiEncoderRanker, CrossEncoderRanker
from bible_search_engine.l2r import L2RRanker, L2RFeatureExtractor
import os
from tqdm import tqdm
import orjson
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
from importlib.resources import files


class BibleSearchEngine:
    def __init__(self, train_mode: bool=False) -> None:
        '''
        Initializes the Bible search engine.
        '''
        # Preprocess Bible data (if needed).
        if train_mode:
            old_testament_path = 'bible_data/old_testament.jsonl'
            new_testament_path = 'bible_data/new_testament.jsonl'

            if not os.path.isfile(old_testament_path) or not os.path.isfile(new_testament_path):
                get_bible_data('kjv-text-files')

            # NLP tokenizer.
            nlp_tokenizer = NLPTokenizer()

            # Bible chapter index.
            bible_chapter_index_path = 'bible_index'

            bible_chapter_index = None
            if not os.path.isdir(bible_chapter_index_path):
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
            verses = {}
            with open(old_testament_path, 'rb') as old_testament_file:
                for bible_chapter_line in tqdm(old_testament_file):
                    bible_chapter = orjson.loads(bible_chapter_line)
                    verses[bible_chapter['chapterid']] = [verse for verse in bible_chapter['verses'].values()]
            with open(new_testament_path, 'rb') as new_testament_file:
                for bible_chapter_line in tqdm(new_testament_file):
                    bible_chapter = orjson.loads(bible_chapter_line)
                    verses[bible_chapter['chapterid']] = [verse for verse in bible_chapter['verses'].values()]
            cross_encoder_ranker = CrossEncoderRanker('cross-encoder/msmarco-MiniLM-L6-en-de-v1',
                                                      verses)

            # Learning to Rank features.
            l2r_feature_extractor = L2RFeatureExtractor(bible_chapter_index, nlp_tokenizer,
                                                        tf_idf_ranker, bm25_ranker,
                                                        dirichlet_lm_ranker, cross_encoder_ranker)

            # Bi-encoder ranker.
            encoded_chapters_path = 'bible_data/encoded_chapters.npy'
            chapter_ids_path = 'bible_data/chapterids.json'

            encoded_chapters = None
            chapter_ids = None
            if not os.path.isfile(encoded_chapters_path):
                bi_encoder_model = SentenceTransformer('msmarco-distilbert-dot-v5', device='cpu')
                chapter_ids = []
                chapter_texts = []
                with open(old_testament_path, 'rb') as old_testament_file:
                    for bible_chapter_line in tqdm(old_testament_file):
                        bible_chapter = orjson.loads(bible_chapter_line)
                        chapter_ids.append(bible_chapter['chapterid'])
                        chapter_texts.append(' '.join(bible_chapter['verses'].values()))
                with open(new_testament_path, 'rb') as new_testament_file:
                    for bible_chapter_line in tqdm(new_testament_file):
                        bible_chapter = orjson.loads(bible_chapter_line)
                        chapter_ids.append(bible_chapter['chapterid'])
                        chapter_texts.append(' '.join(bible_chapter['verses'].values()))
                encoded_chapters = bi_encoder_model.encode(chapter_texts, show_progress_bar=True)
                np.save(encoded_chapters_path, encoded_chapters)
                with open(chapter_ids_path, 'xb') as chapterids_file:
                    chapterids_file.write(orjson.dumps(chapter_ids))
            else:
                encoded_chapters = np.load(encoded_chapters_path)
                with open(chapter_ids_path, 'rb') as chapterids_file:
                    chapter_ids = orjson.loads(chapterids_file.readline())
            bi_encoder_ranker = BiEncoderRanker('msmarco-distilbert-dot-v5', encoded_chapters,
                                                chapter_ids)

            # Learning to Rank ranker.
            self.l2r_ranker = L2RRanker(bible_chapter_index, nlp_tokenizer, bi_encoder_ranker, l2r_feature_extractor)
            self.l2r_ranker.train('bible_queries_relevances/train_queries_relevances.csv')
            joblib.dump(self.l2r_ranker.lightgbm_ranker, "initial_ranker.pkl", True)

            # Bible chapter titles and verses.
            self.chapter_titles = {}
            self.chapter_verses = {}
            with open(old_testament_path, 'rb') as old_testament_file:
                for bible_chapter_line in tqdm(old_testament_file):
                    bible_chapter = orjson.loads(bible_chapter_line)
                    chapterid = bible_chapter['chapterid']
                    self.chapter_titles[chapterid] = bible_chapter['chapter']
                    self.chapter_verses[chapterid] = bible_chapter['verses']
            with open(new_testament_path, 'rb') as new_testament_file:
                for bible_chapter_line in tqdm(new_testament_file):
                    bible_chapter = orjson.loads(bible_chapter_line)
                    chapterid = bible_chapter['chapterid']
                    self.chapter_titles[chapterid] = bible_chapter['chapter']
                    self.chapter_verses[chapterid] = bible_chapter['verses']
        else:
            # NLP tokenizer.
            nlp_tokenizer = NLPTokenizer()

            # Bible chapter index.
            bible_chapter_index_path = files("bible_search_engine.bible_index")
            bible_chapter_index = BibleChapterIndex()
            bible_chapter_index.load(bible_chapter_index_path)

            # Traditional rankers.
            tf_idf_ranker = TFIDFRanker(bible_chapter_index, nlp_tokenizer)
            bm25_ranker = BM25Ranker(bible_chapter_index, nlp_tokenizer)
            dirichlet_lm_ranker = DirichletLMRanker(bible_chapter_index, nlp_tokenizer)

            # Cross-encoder ranker.
            old_testament_path = files("bible_search_engine.bible_data") / 'old_testament.jsonl'
            new_testament_path = files("bible_search_engine.bible_data") / 'new_testament.jsonl'

            verses = {}
            with old_testament_path.open('rb') as old_testament_file:
                for bible_chapter_line in tqdm(old_testament_file):
                    bible_chapter = orjson.loads(bible_chapter_line)
                    verses[bible_chapter['chapterid']] = [verse for verse in bible_chapter['verses'].values()]
            with new_testament_path.open('rb') as new_testament_file:
                for bible_chapter_line in tqdm(new_testament_file):
                    bible_chapter = orjson.loads(bible_chapter_line)
                    verses[bible_chapter['chapterid']] = [verse for verse in bible_chapter['verses'].values()]
            cross_encoder_ranker = CrossEncoderRanker('cross-encoder/msmarco-MiniLM-L6-en-de-v1',
                                                      verses)

            # Learning to Rank features.
            l2r_feature_extractor = L2RFeatureExtractor(bible_chapter_index, nlp_tokenizer,
                                                        tf_idf_ranker, bm25_ranker,
                                                        dirichlet_lm_ranker, cross_encoder_ranker)

            # Bi-encoder ranker.
            encoded_chapters_path = files("bible_search_engine.bible_data") / 'encoded_chapters.npy'
            chapter_ids_path = files("bible_search_engine.bible_data") / 'chapterids.json'

            encoded_chapters = np.load(encoded_chapters_path)
            with chapter_ids_path.open('rb') as chapter_ids_file:
                chapter_ids = orjson.loads(chapter_ids_file.readline())
            bi_encoder_ranker = BiEncoderRanker('msmarco-distilbert-dot-v5', encoded_chapters,
                                                chapter_ids)

            # Learning to Rank ranker.
            self.l2r_ranker = L2RRanker(bible_chapter_index, nlp_tokenizer, bi_encoder_ranker, l2r_feature_extractor)
            self.l2r_ranker.lightgbm_ranker = joblib.load(files("bible_search_engine") / "initial_ranker.pkl")

            # Bible chapter titles and verses.
            self.chapter_titles = {}
            self.chapter_verses = {}
            with old_testament_path.open('rb') as old_testament_file:
                for bible_chapter_line in tqdm(old_testament_file):
                    bible_chapter = orjson.loads(bible_chapter_line)
                    chapterid = bible_chapter['chapterid']
                    self.chapter_titles[chapterid] = bible_chapter['chapter']
                    self.chapter_verses[chapterid] = bible_chapter['verses']
            with new_testament_path.open('rb') as new_testament_file:
                for bible_chapter_line in tqdm(new_testament_file):
                    bible_chapter = orjson.loads(bible_chapter_line)
                    chapterid = bible_chapter['chapterid']
                    self.chapter_titles[chapterid] = bible_chapter['chapter']
                    self.chapter_verses[chapterid] = bible_chapter['verses']

    def search(self, query: str) -> list[dict]:
        '''
        Search for the query with the Bible search engine

        query: Query of interest.

        Returns search results, including chapter id, chapter score, chapter title,
        and chapter verses, for the query with the Bible search engine
        '''
        query_results = self.l2r_ranker.query(query)
        search_results = [{'chapterid': chapterid, 'score': score, 'chapter': self.chapter_titles[chapterid],
                           'verses': self.chapter_verses[chapterid]}
                          for chapterid, score in query_results]
        return search_results

def create_bible_search_engine(train_mode: bool=False) -> BibleSearchEngine:
    '''
    Creates the Bible search engine.

    Returns the created Bible search engine.
    '''
    return BibleSearchEngine(train_mode)
