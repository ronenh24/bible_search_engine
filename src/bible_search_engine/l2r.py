# Author: Ronen H

from bible_search_engine.indexing import BibleChapterIndex
from bible_search_engine.document_preprocessor import NLPTokenizer
import lightgbm
import multiprocessing
import os
import pandas as pd
from tqdm import tqdm
from bible_search_engine.ranker import TFIDFRanker, BM25Ranker, DirichletLMRanker, CrossEncoderRanker
import numpy as np


class L2RRanker:
    def __init__(self, bible_chapter_index: BibleChapterIndex, nlp_tokenizer: NLPTokenizer,
                 ranker, feature_extractor: 'L2RFeatureExtractor',
                 lgbmranker_params: dict | None = None) -> None:
        '''
        Initializes learning to rank with LightGBM ranker.

        bible_chapter_index: Bible chapter index.
        nlp_tokenizer: NLP tokenizer.
        ranker: Ranker with query.
        feature_extractor: Feature extractor for LightGBM ranker.
        lgbmranker_params: Parameters for LightGBM ranker.
        '''
        self.bible_chapter_index = bible_chapter_index
        self.nlp_tokenizer = nlp_tokenizer
        self.ranker = ranker
        self.feature_extractor = feature_extractor
        initial_lgbmranker_params = {'num_leaves': 10, 'learning_rate': 0.005, 'n_estimators': 50,
                                     'n_jobs': multiprocessing.cpu_count(), 'importance_type': 'gain',
                                     'metric': 'ndcg', 'verbosity': 1}
        if lgbmranker_params:
            initial_lgbmranker_params.update(lgbmranker_params)
        self.lightgbm_ranker = lightgbm.LGBMRanker().set_params(**initial_lgbmranker_params)
    
    def train(self, train_queries_data: str) -> None:
        '''
        Train LightGBM ranker with train queries and their relevance scores.
        '''
        if not os.path.isfile(train_queries_data):
            raise Exception('Train queries data does not exist.')
        
        train_queries_df = pd.read_csv(train_queries_data)

        train_features = []
        train_relevance_scores = []
        train_num_query_examples = []

        for query in tqdm(train_queries_df['query'].unique()):
            train_query_df = train_queries_df[train_queries_df['query'] == query][['chapterid', 'relevance']]
            train_num_query_examples.append(train_query_df.shape[0])
            query_parts = self.nlp_tokenizer.tokenize(query)
            set_query_parts = set([query_part for query_part in query_parts if query_part])
            for chapterid, relevance_score in train_query_df.itertuples(index=False, name=None):
                chapter_term_counts = {}
                for query_part in set_query_parts.intersection(self.bible_chapter_index.get_chapter_vocab(chapterid)):
                    chapter_term_counts[query_part] = self.bible_chapter_index.get_chapter_term_freq(chapterid,
                                                                                                     query_part)
                train_features.append(self.feature_extractor.get_features(chapterid, chapter_term_counts,
                                                                          query_parts, query))
                train_relevance_scores.append(relevance_score)
        
        self.lightgbm_ranker.fit(train_features, train_relevance_scores, group=train_num_query_examples)
    
    def query(self, query: str) -> list[tuple[int, float]]:
        '''
        Search for the relevant Bible chapters to the query.

        query: Query of interest.

        Returns the most relevant Bible chapters for the query.
        '''
        query_parts = self.nlp_tokenizer.tokenize(query)
        set_query_parts = set([query_part for query_part in query_parts if query_part])

        initial_results = self.ranker.query(query)
        if initial_results == []:
            return initial_results
        
        # Rerank top 50 initial results with learning to rank.
        top_100_initial_results = initial_results[:100]
        test_features = []
        for chapterid, _ in tqdm(top_100_initial_results):
            chapter_term_counts = {}
            for query_part in set_query_parts.intersection(self.bible_chapter_index.get_chapter_vocab(chapterid)):
                chapter_term_counts[query_part] = self.bible_chapter_index.get_chapter_term_freq(chapterid, query_part)
            test_features.append(self.feature_extractor.get_features(chapterid, chapter_term_counts,
                                                                     query_parts, query))
        
        results = [top_100_initial_results[i] for i in np.argsort(self.lightgbm_ranker.predict(test_features))[::-1]]
        results.extend(initial_results[100:])

        return results
    
    def ranker_name(self) -> str:
        return 'l2r_ranker'


class L2RFeatureExtractor:
    def __init__(self, bible_chapter_index: BibleChapterIndex, nlp_tokenizer: NLPTokenizer,
                 tf_idf_ranker: TFIDFRanker, bm25_ranker: BM25Ranker,
                 dirichlet_lm_ranker: DirichletLMRanker, cross_encoder_ranker: CrossEncoderRanker) -> None:
        '''
        Initializes feature extractor for learning to rank.

        bible_chapter_index: Bible chapter index.
        nlp_tokenizer: NLP tokenizer.
        tf_idf_ranker: TF-IDF ranker with score.
        bm25_ranker: BM25 ranker with score.
        dirichlet_lm_ranker: Dirichlet LM ranker with score.
        cross_encoder_ranker: Cross-encoder ranker with score.
        '''
        self.bible_chapter_index = bible_chapter_index
        self.nlp_tokenizer = nlp_tokenizer
        self.tf_idf_ranker = tf_idf_ranker
        self.bm25_ranker = bm25_ranker
        self.dirichlet_lm_ranker = dirichlet_lm_ranker
        self.cross_encoder_ranker = cross_encoder_ranker
    
    def get_number_verses(self, chapterid: int) -> int:
        '''
        Gets number of verses for the Bible chapter.

        chapterid: Bible chapter id.

        Returns number of verses for the Bible chapter.
        '''
        bible_chapter_metadata = self.bible_chapter_index.get_chapter_metadata(chapterid)
        if bible_chapter_metadata == {}:
            return 0
        return bible_chapter_metadata['number_of_verses']
    
    def get_chapter_length(self, chapterid: int) -> int:
        '''
        Gets length of the Bible chapter.

        chapterid: Bible chapter id.

        Returns length of the Bible chapter.
        '''
        bible_chapter_metadata = self.bible_chapter_index.get_chapter_metadata(chapterid)
        if bible_chapter_metadata == {}:
            return 0
        return bible_chapter_metadata['chapter_length']
    
    def get_tf_idf(self, chapterid: int, chapter_term_counts: dict[str, int], query_parts: list[str]) -> float:
        '''
        Gets TF-IDF score of the Bible chapter.

        chapterid: Bible chapter id.
        chapter_term_counts: Count of each term in Bible chapter.
        query_parts: Tokenized query.

        Returns TF-IDF score of the Bible chapter.
        '''
        return self.tf_idf_ranker.score(chapterid, chapter_term_counts, query_parts)
    
    def get_bm25(self, chapterid: int, chapter_term_counts: dict[str, int], query_parts: list[str]) -> float:
        '''
        Gets BM25 score of the Bible chapter.

        chapterid: Bible chapter id.
        chapter_term_counts: Count of each term in Bible chapter.
        query_parts: Tokenized query.

        Returns BM25 score of the Bible chapter.
        '''
        return self.bm25_ranker.score(chapterid, chapter_term_counts, query_parts)
    
    def get_dirichlet_lm(self, chapterid: int, chapter_term_counts: dict[str, int], query_parts: list[str]) -> float:
        '''
        Gets Dirichlet LM score of the Bible chapter.

        chapterid: Bible chapter id.
        chapter_term_counts: Count of each term in Bible chapter.
        query_parts: Tokenized query.

        Returns Dirichlet LM score of the Bible chapter.
        '''
        return self.dirichlet_lm_ranker.score(chapterid, chapter_term_counts, query_parts)
    
    def get_cross_encoder_score(self, chapterid: int, query: str) -> float:
        '''
        Gets cross-encoder score of the Bible chapter.

        chapterid: Bible chapter id.
        query: Query of interest.

        Returns cross-encoder score of the Bible chapter.
        '''
        return self.cross_encoder_ranker.score(chapterid, query)
    
    def get_features(self, chapterid: int, chapter_term_counts: dict[str, int],
                     query_parts: list[str], query: str) -> list[float]:
        '''
        Get features of the Bible chapter.

        chapterid: Bible chapter id.
        chapter_term_counts: Count of each term in Bible chapter.
        query_parts: Tokenized query.
        query: Query of interest.

        Returns features of the Bible chapter as a vector.
        '''
        bible_chapter_features = []

        # Number of verses.
        bible_chapter_features.append(self.get_number_verses(chapterid))

        # Bible chapter Length.
        bible_chapter_features.append(self.get_chapter_length(chapterid))

        # TF-IDF.
        bible_chapter_features.append(self.get_tf_idf(chapterid, chapter_term_counts, query_parts))

        # BM25.
        bible_chapter_features.append(self.get_bm25(chapterid, chapter_term_counts, query_parts))

        # Dirichlet LM.
        bible_chapter_features.append(self.get_dirichlet_lm(chapterid, chapter_term_counts, query_parts))

        # Cross-encoder.
        bible_chapter_features.append(self.get_cross_encoder_score(chapterid, query))

        return bible_chapter_features

