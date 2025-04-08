# Author: Ronen Huang

from bible_search_engine.components.indexing import BibleChapterIndex
from bible_search_engine.components.preprocess import NLPTokenizer
from bible_search_engine.components.ranker import (TraditionalRanker, TFIDFRanker, BM25Ranker, DirichletLMRanker,
                                                   BiEncoderRanker, CrossEncoderRanker, ColbertRanker)
from bible_search_engine.components.relevance import Relevance
import lightgbm
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import optuna
import multiprocessing
import scipy.stats as st


class L2RRanker:
    """
    Learning to Rank Bible chapter retrieval.
    """
    def __init__(self, bible_chapter_index: BibleChapterIndex, nlp_tokenizer: NLPTokenizer,
                 ranker: TraditionalRanker | BiEncoderRanker, feature_extractor: 'L2RFeatureExtractor') -> None:
        '''
        bible_chapter_index: Bible chapter index.
        nlp_tokenizer: NLP tokenizer.
        ranker: Ranker with query.
        feature_extractor: Feature extractor for LightGBM ranker.

        Initializes learning to rank with LightGBM ranker.
        '''
        self.bible_chapter_index = bible_chapter_index
        self.nlp_tokenizer = nlp_tokenizer
        self.ranker = ranker
        self.feature_extractor = feature_extractor
        self.lightgbm_ranker = None
    
    def train(self, train_queries_data: str, test_queries_data: str) -> None:
        '''
        Train LightGBM ranker with train queries and their relevance scores.
        '''
        if not os.path.isfile(train_queries_data):
            raise Exception('Train queries data does not exist.')
        if not os.path.isfile(test_queries_data):
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

        test_eval = Relevance(test_queries_data)

        params_study = optuna.create_study(study_name="Bible Search Engine Ranker Parameter Optimization",
                                           direction="maximize")
        params_study.optimize(lambda trial: self.determine_params(trial, train_features, train_relevance_scores,
                                                                  train_num_query_examples, test_eval),
                              10)
        best_params = params_study.best_params
        lgbmranker_params = {'num_leaves': best_params["num_leaves"], 'learning_rate': best_params["learning_rate"],
                             'max_depth': best_params["max_depth"], 'num_iterations': best_params["num_iterations"],
                             'min_data_in_leaf': best_params["min_data_in_leaf"], 'metric': 'ndcg',
                             'importance_type': 'gain', 'verbosity': 2, 'n_jobs': multiprocessing.cpu_count()}
        self.lightgbm_ranker = lightgbm.LGBMRanker().set_params(**lgbmranker_params)
        self.lightgbm_ranker.fit(train_features, train_relevance_scores, group=train_num_query_examples)

    def determine_params(self, trial: optuna.Trial, train_features: list[list[float]],
                         train_relevance_scores: list[float], train_num_query_examples: list[int],
                         test_eval: Relevance):
        learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True)
        num_iterations = trial.suggest_int("num_iterations", 1000, 2000, log=True)
        max_depth = trial.suggest_int("max_depth", 7, 14)
        num_leaves = trial.suggest_int("num_leaves", 2 ** (max_depth - 1), 2 ** max_depth)
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 5, 35)

        lgbmranker_params = {'num_leaves': num_leaves, 'learning_rate': learning_rate, 'max_depth': max_depth,
                             'num_iterations': num_iterations, 'min_data_in_leaf': min_data_in_leaf, 'metric': 'ndcg',
                             'importance_type': 'gain', 'verbosity': 2, 'n_jobs': multiprocessing.cpu_count()}
        self.lightgbm_ranker = lightgbm.LGBMRanker().set_params(**lgbmranker_params)
        self.lightgbm_ranker.fit(train_features, train_relevance_scores, group=train_num_query_examples)

        results = [s for _, s in test_eval.evaluate_ranker_results(self)]
        return st.t.interval(0.95, len(results) - 1, np.mean(results), st.sem(results))[0]
    
    def query(self, query: str) -> list[tuple[int, float]]:
        '''
        query: Query of interest.

        Search for the relevant Bible chapters to the query.
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
    """
    Learning to Rank feature extraction.
    """
    def __init__(self, bible_chapter_index: BibleChapterIndex, nlp_tokenizer: NLPTokenizer,
                 tf_idf_ranker: TFIDFRanker, bm25_ranker: BM25Ranker,
                 dirichlet_lm_ranker: DirichletLMRanker, colbert_ranker: ColbertRanker) -> None:
        '''
        bible_chapter_index: Bible chapter index.
        nlp_tokenizer: NLP tokenizer.
        tf_idf_ranker: TF-IDF ranker with score.
        bm25_ranker: BM25 ranker with score.
        dirichlet_lm_ranker: Dirichlet LM ranker with score.
        colbert_ranker: Colbert ranker with score.
        '''
        self.bible_chapter_index = bible_chapter_index
        self.nlp_tokenizer = nlp_tokenizer
        self.tf_idf_ranker = tf_idf_ranker
        self.bm25_ranker = bm25_ranker
        self.dirichlet_lm_ranker = dirichlet_lm_ranker
        #self.cross_encoder_ranker = cross_encoder_ranker
        self.colbert_ranker = colbert_ranker
    
    def get_number_verses(self, chapterid: int) -> int:
        '''
        chapterid: Bible chapter id.

        Gets number of verses for the Bible chapter.
        '''
        bible_chapter_metadata = self.bible_chapter_index.get_chapter_metadata(chapterid)
        if bible_chapter_metadata == {}:
            return 0
        return bible_chapter_metadata['number_of_verses']
    
    def get_chapter_length(self, chapterid: int) -> int:
        '''
        chapterid: Bible chapter id.

        Gets length of the Bible chapter.
        '''
        bible_chapter_metadata = self.bible_chapter_index.get_chapter_metadata(chapterid)
        if bible_chapter_metadata == {}:
            return 0
        return bible_chapter_metadata['chapter_length']
    
    def get_tf_idf(self, chapterid: int, chapter_term_counts: dict[str, int], query_parts: list[str]) -> float:
        '''
        chapterid: Bible chapter id.
        chapter_term_counts: Count of each term in Bible chapter.
        query_parts: Tokenized query.

        Gets TF-IDF score of the Bible chapter.
        '''
        return self.tf_idf_ranker.score(chapterid, chapter_term_counts, query_parts)
    
    def get_bm25(self, chapterid: int, chapter_term_counts: dict[str, int], query_parts: list[str]) -> float:
        '''
        chapterid: Bible chapter id.
        chapter_term_counts: Count of each term in Bible chapter.
        query_parts: Tokenized query.

        Gets BM25 score of the Bible chapter.
        '''
        return self.bm25_ranker.score(chapterid, chapter_term_counts, query_parts)
    
    def get_dirichlet_lm(self, chapterid: int, chapter_term_counts: dict[str, int], query_parts: list[str]) -> float:
        '''
        chapterid: Bible chapter id.
        chapter_term_counts: Count of each term in Bible chapter.
        query_parts: Tokenized query.

        Gets Dirichlet LM score of the Bible chapter.
        '''
        return self.dirichlet_lm_ranker.score(chapterid, chapter_term_counts, query_parts)
    
    #def get_cross_encoder_score(self, chapterid: int, query: str) -> float:
    #    '''

    #    chapterid: Bible chapter id.
    #    query: Query of interest.

    #    Gets cross-encoder score of the Bible chapter.
    #    '''
    #    return self.cross_encoder_ranker.score(chapterid, query)

    def get_colbert_score(self, chapterid: int, query: str) -> float:
        """
        chapterid: Bible chapter id.
        query: Query of interest.

        Gets colbert score of the Bible chapter.
        """
        return self.colbert_ranker.score(chapterid, query)
    
    def get_features(self, chapterid: int, chapter_term_counts: dict[str, int],
                     query_parts: list[str], query: str) -> list[float]:
        '''
        chapterid: Bible chapter id.
        chapter_term_counts: Count of each term in Bible chapter.
        query_parts: Tokenized query.
        query: Query of interest.

        Get features of the Bible chapter.
        '''
        bible_chapter_features = []

        # Chapter ID.
        bible_chapter_features.append(chapterid)

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
        #bible_chapter_features.append(self.get_cross_encoder_score(chapterid, query))

        # Colbert.
        bible_chapter_features.append(self.get_colbert_score(chapterid, query))

        return bible_chapter_features
