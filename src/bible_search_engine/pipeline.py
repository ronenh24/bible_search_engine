"""
Author: Ronen Huang
"""

from importlib.resources import files
from importlib.resources.abc import Traversable
import os
from typing import Literal
# from dotenv import load_dotenv
from tqdm import tqdm, trange
import orjson
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
from bible_search_engine.components.get_bible_data import get_bible_data
from bible_search_engine.components.preprocess import NLPTokenizer
from bible_search_engine.components.indexing import\
    BibleChapterIndex, create_bible_index
from bible_search_engine.components.ranker import\
    TFIDFRanker, BM25Ranker, DirichletLMRanker, \
    BiEncoderRanker, CrossEncoderRanker, ColbertRanker
from bible_search_engine.components.l2r import L2RRanker, L2RFeatureExtractor


# load_dotenv()


class BibleSearchEngine:
    """
    Bible Search Engine with search method.
    """
    def __init__(self, version: Literal["csv", "esv", "kjv", "msg", "nas",
                                        "niv", "nkjv", "nlt", "nrs"] = "niv",
                 train_mode: bool = False, keep: set[str] = None) -> None:
        '''
        Initializes the Bible search engine.

        For training, should be run from src directory.
        '''
        if keep is None:
            keep = set()

        if not train_mode:
            keep = set(["get_bible_data", "indexing", "colbert", "biencoder"])

        old_testament_file =\
            'old_testament_' + version + '.jsonl'
        old_testament_path =\
            files("bible_search_engine.bible_data") / old_testament_file
        new_testament_file =\
            'new_testament_' + version + '.jsonl'
        new_testament_path =\
            files("bible_search_engine.bible_data") / new_testament_file
        bible_chapter_index_path = files(
            "bible_search_engine.bible_index_" + version
        )
        traditional_hyperparams_file =\
            "initial_params_" + version + ".json"
        traditional_hyperparams_path =\
            files("bible_search_engine") / traditional_hyperparams_file
        encoded_chapters_file = 'encoded_chapters_' + version + '.npy'
        encoded_chapters_path =\
            files("bible_search_engine.bible_data") / encoded_chapters_file
        chapter_ids_file = 'chapterids_' + version + '.json'
        chapter_ids_path =\
            files("bible_search_engine.bible_data") / chapter_ids_file
        train_queries_path =\
            files('bible_search_engine.bible_queries_relevances') /\
            'train_queries_relevances.csv'
        test_queries_path =\
            files('bible_search_engine.bible_queries_relevances') /\
            'test_queries_relevances.csv'
        initial_ranker_file =\
            "initial_ranker_" + version + ".pkl"
        initial_ranker_path =\
            files("bible_search_engine") / initial_ranker_file

        # NLP tokenizer.
        nlp_tokenizer = NLPTokenizer()

        # Preprocess Bible data.
        if "get_bible_data" not in keep:
            get_bible_data(version)

        # Bible chapter index.
        if "indexing" not in keep:
            bible_chapter_index = create_bible_index(
                str(old_testament_path),
                str(new_testament_path),
                nlp_tokenizer
            )
            bible_chapter_index.save(str(bible_chapter_index_path))
        else:
            bible_chapter_index = BibleChapterIndex()
            bible_chapter_index.load(bible_chapter_index_path)

        # Traditional rankers.
        tf_idf_ranker = TFIDFRanker(bible_chapter_index, nlp_tokenizer)
        bm25_ranker = BM25Ranker(bible_chapter_index, nlp_tokenizer)
        dirichlet_lm_ranker = DirichletLMRanker(
            bible_chapter_index, nlp_tokenizer
        )
        if not train_mode:
            with traditional_hyperparams_path.open("rb") as hp_file:
                hp_dict = orjson.loads(hp_file.readline())
            bm25_ranker.set_params(hp_dict["b"], hp_dict["k1"], hp_dict["k3"])
            dirichlet_lm_ranker.set_params(hp_dict["mu"])

        # Cross-encoder ranker.
        # cross_encoder_ranker = self.get_cross_encoder(
        #   old_testament_path.open('rb'), new_testament_path.open('rb')
        # )

        # Bible chapter titles and verses.
        chapter_ids, chapter_texts = self.get_chapter_verses(
            old_testament_path, new_testament_path
        )

        # Colbert ranker.
        if "colbert" not in keep:
            self.get_colbert(chapter_ids, chapter_texts)
        colbert_ranker = ColbertRanker(
            "colbert-ir/colbertv2.0", "bible_chapters"
        )

        # Learning to Rank features.
        l2r_feature_extractor = L2RFeatureExtractor(
            bible_chapter_index, nlp_tokenizer, tf_idf_ranker,
            bm25_ranker, dirichlet_lm_ranker, colbert_ranker
        )

        # Bi-encoder ranker.
        bi_encoder_model = SentenceTransformer(
            'msmarco-distilbert-dot-v5', device='cpu'
        )
        if "biencoder" not in keep:
            encoded_chapters = bi_encoder_model.encode(
                chapter_texts, show_progress_bar=True
            )
            np.save(str(encoded_chapters_path), encoded_chapters)
            with open(str(chapter_ids_path), 'wb') as chapterids_file:
                chapterids_file.write(orjson.dumps(chapter_ids))
        else:
            encoded_chapters = np.load(encoded_chapters_path)
            with chapter_ids_path.open("rb") as chapter_ids_file:
                chapter_ids = orjson.loads(chapter_ids_file.readline())
        bi_encoder_ranker = BiEncoderRanker(
            'msmarco-distilbert-dot-v5', encoded_chapters, chapter_ids
        )

        # Learning to Rank ranker.
        self.l2r_ranker = L2RRanker(
            bible_chapter_index, nlp_tokenizer,
            bi_encoder_ranker, l2r_feature_extractor
        )
        if train_mode:
            self.l2r_ranker.train(
                str(train_queries_path), str(test_queries_path)
            )
            joblib.dump(
                self.l2r_ranker.lightgbm_ranker, str(initial_ranker_path), True
            )
            trained_bm25_ranker = self.l2r_ranker.feature_extractor.bm25_ranker
            b, k1, k3 = trained_bm25_ranker.get_params()
            trained_dirichlet_lm_ranker =\
                self.l2r_ranker.feature_extractor.dirichlet_lm_ranker
            mu = trained_dirichlet_lm_ranker.get_params()
            with open(str(traditional_hyperparams_path), 'wb') as hp_file:
                hp_file.write(
                    orjson.dumps(
                        {"b": b, "k1": k1, "k3": k3, "mu": mu}
                    )
                )
        else:
            self.l2r_ranker.lightgbm_ranker = joblib.load(
                initial_ranker_path
            )

    def get_cross_encoder(self, old_testament_path: Traversable,
                          new_testament_path: Traversable) ->\
            CrossEncoderRanker:
        verses = {}
        with old_testament_path.open("rb") as old_testament_file:
            for bible_chapter_line in tqdm(old_testament_file):
                bible_chapter = orjson.loads(bible_chapter_line)
                verses[bible_chapter['chapterid']] =\
                    [verse for verse in bible_chapter['verses'].values()]
        with new_testament_path.open("rb") as new_testament_file:
            for bible_chapter_line in tqdm(new_testament_file):
                bible_chapter = orjson.loads(bible_chapter_line)
                verses[bible_chapter['chapterid']] =\
                    [verse for verse in bible_chapter['verses'].values()]
        return CrossEncoderRanker(
            'cross-encoder/msmarco-MiniLM-L6-en-de-v1', verses
        )

    def get_colbert(self, chapter_ids: list[int], chapter_texts: list[str]) ->\
            None:
        colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
        chapter_embeddings = list(colbert_model.embed(chapter_texts))
        colbert_client = QdrantClient(
            url=os.environ.get("url"), api_key=os.environ.get("api_key")
        )
        colbert_client.create_collection(
            collection_name="bible_chapters",
            vectors_config=models.VectorParams(
                size=128, distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                )
            )
        )
        for i in trange(0, len(chapter_ids), 10):
            start = i
            end = min(len(chapter_ids), i + 10)
            colbert_client.upsert(
                collection_name="bible_chapters",
                points=[models.PointStruct(
                            id=chapter_id, vector=chapter_embedding
                        ) for chapter_id, chapter_embedding in
                        zip(
                            chapter_ids[start:end],
                            chapter_embeddings[start:end]
                            )
                        ]
            )

    def get_chapter_verses(self, old_testament_path: Traversable,
                           new_testament_path: Traversable) ->\
            tuple[list[int], list[str]]:
        self.chapter_titles = {}
        self.chapter_verses = {}
        chapter_ids = []
        chapter_texts = []
        with old_testament_path.open("rb") as old_testament_file:
            for bible_chapter_line in tqdm(old_testament_file):
                bible_chapter = orjson.loads(bible_chapter_line)
                chapterid = bible_chapter['chapterid']
                self.chapter_titles[chapterid] = bible_chapter['chapter']
                self.chapter_verses[chapterid] = bible_chapter['verses']
                chapter_ids.append(chapterid)
                chapter_texts.append(
                    " ".join(bible_chapter["verses"].values())
                )
        with new_testament_path.open("rb") as new_testament_file:
            for bible_chapter_line in tqdm(new_testament_file):
                bible_chapter = orjson.loads(bible_chapter_line)
                chapterid = bible_chapter['chapterid']
                self.chapter_titles[chapterid] = bible_chapter['chapter']
                self.chapter_verses[chapterid] = bible_chapter['verses']
                chapter_ids.append(chapterid)
                chapter_texts.append(
                    " ".join(bible_chapter["verses"].values())
                )
        return chapter_ids, chapter_texts

    def search(self, query: str) -> list[dict]:
        '''
        query: Query of interest.

        Search for the query with the Bible search engine.
        '''
        query_results = self.l2r_ranker.query(query)
        search_results = [
            {
                'chapterid': chapterid, 'score': score,
                'chapter': self.chapter_titles[chapterid],
                'verses': self.chapter_verses[chapterid]
            }
            for chapterid, score in query_results]
        return search_results


def create_bible_search_engine(version: str = "niv", train_mode: bool = False,
                               keep: set[str] = None) -> BibleSearchEngine:
    '''
    Creates the Bible search engine.

    Returns the created Bible search engine.
    '''
    return BibleSearchEngine(version, train_mode, keep)
