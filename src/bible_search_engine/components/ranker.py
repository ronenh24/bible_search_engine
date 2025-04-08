# Author: Ronen Huang

from bible_search_engine.components.indexing import BibleChapterIndex
from bible_search_engine.components.preprocess import NLPTokenizer
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
from collections import Counter
from tqdm import tqdm
from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient


class TraditionalRanker():
    """
    Base class for traditional rankers.
    """
    def __init__(self, bible_chapter_index: BibleChapterIndex, nlp_tokenizer: NLPTokenizer):
        '''
        bible_chapter_index: Bible chapter index.
        nlp_tokenizer: NLP tokenizer.

        Initializes the traditional ranker.
        '''
        self.bible_chapter_index = bible_chapter_index
        self.nlp_tokenizer = nlp_tokenizer

    def query(self, query: str) -> list[tuple[int, int]]:
        '''
        query: Query of interest.

        Search for the relevant Bible chapters to the query.
        '''
        pass

    def score(self, chapterid: int, query_parts: list[str]) -> int:
        '''
        chapterid: Bible chapter id.
        query_parts: Tokenized query.

        Scores the relevance of the Bible chapter to the query.
        '''
        pass

    def ranker_name(self) -> str:
        pass


class RandomRanker(TraditionalRanker):
    """
    Shortest length ranking of relevant Bible chapters.
    """
    def __init__(self, bible_chapter_index: BibleChapterIndex, nlp_tokenizer: NLPTokenizer) -> None:
        super().__init__(bible_chapter_index, nlp_tokenizer)

    def query(self, query: str) -> list[tuple[int, int]]:
        results = []
        
        query_parts = self.nlp_tokenizer.tokenize(query)

        for chapterid in tqdm(self.bible_chapter_index.get_chapter_ids()):
            score = self.score(chapterid, query_parts)
            if score != 0:
                results.append((chapterid, score))

        results = sorted(results, key=lambda chapter_score: chapter_score[1], reverse=True)

        return results

    def score(self, chapterid: int, query_parts: list[str]) -> int:
        """
        Inverse Bible chapter length.
        """
        score = 0

        set_query_parts = set([query_part for query_part in query_parts if query_part])
        chapter_vocab = self.bible_chapter_index.get_chapter_vocab(chapterid)

        if len(set_query_parts.intersection(chapter_vocab)) > 0:
            score = 1 / self.bible_chapter_index.get_chapter_metadata(chapterid)['chapter_length']
        
        return score
    
    def ranker_name(self) -> str:
        return 'random_ranker'


class TFIDFRanker(TraditionalRanker):
    """
    TF-IDF ranking of relevant Bible chapters.
    """
    def __init__(self, bible_chapter_index: BibleChapterIndex, nlp_tokenizer: NLPTokenizer) -> None:
        super().__init__(bible_chapter_index, nlp_tokenizer)
    
    def query(self, query: str) -> list[tuple[int, int]]:
        results = []
        
        query_parts = self.nlp_tokenizer.tokenize(query)
        set_query_parts = set([query_part for query_part in query_parts if query_part])

        for chapterid in tqdm(self.bible_chapter_index.get_chapter_ids()):
            chapter_term_counts = {}
            for query_part in set_query_parts.intersection(self.bible_chapter_index.get_chapter_vocab(chapterid)):
                chapter_term_counts[query_part] = self.bible_chapter_index.get_chapter_term_freq(chapterid, query_part)
            score = self.score(chapterid, chapter_term_counts, query_parts)
            if score != 0:
                results.append((chapterid, score))

        results = sorted(results, key=lambda chapter_score: chapter_score[1], reverse=True)

        return results

    def score(self, chapterid: int, chapter_term_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        TF-IDF.
        """
        number_of_chapters = self.bible_chapter_index.get_statistics()['number_of_chapters']

        score = 0

        set_query_parts = set([query_part for query_part in query_parts if query_part])

        for query_part in set_query_parts:
            if query_part in chapter_term_counts:
                c_t_d = chapter_term_counts[query_part]
                df_t = self.bible_chapter_index.get_term_metadata(query_part)['number_of_chapters_appears']
                idf = 1 + np.log(number_of_chapters / df_t)
                tf = np.log(c_t_d + 1)
                score += idf * tf

        return score
    
    def ranker_name(self) -> str:
        return 'tf_idf_ranker'


class BM25Ranker(TraditionalRanker):
    """
    BM25 ranking of relevant Bible chapters.
    """
    def __init__(self, bible_chapter_index: BibleChapterIndex, nlp_tokenizer: NLPTokenizer,
                 b: float = 0.75, k1: float = 1.2, k3: float = 8) -> None:
        super().__init__(bible_chapter_index, nlp_tokenizer)
        self.b = b
        self.k1 = k1
        self.k3 = k3

    def query(self, query: str) -> list[tuple[int, int]]:
        results = []
        
        query_parts = self.nlp_tokenizer.tokenize(query)
        set_query_parts = set([query_part for query_part in query_parts if query_part])

        for chapterid in tqdm(self.bible_chapter_index.get_chapter_ids()):
            chapter_term_counts = {}
            for query_part in set_query_parts.intersection(self.bible_chapter_index.get_chapter_vocab(chapterid)):
                chapter_term_counts[query_part] = self.bible_chapter_index.get_chapter_term_freq(chapterid, query_part)
            score = self.score(chapterid, chapter_term_counts, query_parts)
            if score != 0:
                results.append((chapterid, score))

        results = sorted(results, key=lambda chapter_score: chapter_score[1], reverse=True)

        return results

    def score(self, chapterid: int, chapter_term_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        BM25.
        """
        bible_statistics = self.bible_chapter_index.get_statistics()
        chapter_statistics = self.bible_chapter_index.get_chapter_metadata(chapterid)

        number_of_chapters = bible_statistics['number_of_chapters']
        chapter_length = chapter_statistics['chapter_length']
        mean_chapter_length = bible_statistics['mean_chapter_length']

        query_parts_counts = Counter([query_part for query_part in query_parts if query_part])

        score = 0

        for query_part, c_t_q in query_parts_counts.items():
            if query_part in chapter_term_counts:
                c_t_d = chapter_term_counts[query_part]
                df_t = self.bible_chapter_index.get_term_metadata(query_part)['number_of_chapters_appears']
                idf = np.log((number_of_chapters - df_t + 0.5) / (df_t + 0.5))
                tf = ((self.k1 + 1) * c_t_d) /\
                     (self.k1 * (1 - self.b + self.b * (chapter_length / mean_chapter_length)) + c_t_d)
                qtf = ((self.k3 + 1) * c_t_q) / (self.k3 + c_t_q)
                score += idf * tf * qtf

        return score
    
    def ranker_name(self) -> str:
        return 'bm25_ranker'


class DirichletLMRanker(TraditionalRanker):
    """
    Dirichlet LM ranking of relevant Bible chapters.
    """
    def __init__(self, bible_chapter_index: BibleChapterIndex, nlp_tokenizer: NLPTokenizer,
                 mu: float = 2000) -> None:
        super().__init__(bible_chapter_index, nlp_tokenizer)
        self.mu = mu
    
    def query(self, query: str) -> list[tuple[int, int]]:
        results = []
        
        query_parts = self.nlp_tokenizer.tokenize(query)
        set_query_parts = set([query_part for query_part in query_parts if query_part])

        for chapterid in tqdm(self.bible_chapter_index.get_chapter_ids()):
            chapter_term_counts = {}
            for query_part in set_query_parts.intersection(self.bible_chapter_index.get_chapter_vocab(chapterid)):
                chapter_term_counts[query_part] = self.bible_chapter_index.get_chapter_term_freq(chapterid, query_part)
            score = self.score(chapterid, chapter_term_counts, query_parts)
            if score != 0:
                results.append((chapterid, score))

        results = sorted(results, key=lambda chapter_score: chapter_score[1], reverse=True)

        return results

    def score(self, chapterid: int, chapter_term_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Dirichlet LM
        """
        bible_statistics = self.bible_chapter_index.get_statistics()
        chapter_statistics = self.bible_chapter_index.get_chapter_metadata(chapterid)

        total_token_count = bible_statistics['total_token_count']
        chapter_length = chapter_statistics['chapter_length']
        query_len = len(query_parts)

        query_parts_counts = Counter([query_part for query_part in query_parts if query_part])

        score = 0

        for query_part, c_t_q in query_parts_counts.items():
            if query_part in chapter_term_counts:
                c_t_d = chapter_term_counts[query_part]
                p_w_c = self.bible_chapter_index.get_term_metadata(query_part)['total_term_frequency'] /\
                        total_token_count
                ml_estimate = c_t_q * np.log(1 + (c_t_d / (self.mu * p_w_c)))
                score += ml_estimate
        
        if score > 0:
            param = query_len * np.log(self.mu / (chapter_length + self.mu))
            score += param
        
        return score
    
    def ranker_name(self) -> str:
        return 'dirichlet_lm_ranker'


class BiEncoderRanker:
    """
    Bi-encoder ranker.
    """
    def __init__(self, bi_encoder_model_name: str, encoded_chapters: np.ndarray, chapter_ids: list[int]) -> None:
        '''
        bi_encoder_model_name: Bi-encoder model.
        encoded_chapters: Encoded Bible chapters from the bi-encoder model.
        chapter_ids: Bible chapter ids of the encoded Bible chapters.

        Initializes the bi-encoder ranker.
        '''
        self.bi_encoder_model = SentenceTransformer(bi_encoder_model_name)
        self.encoded_chapters = encoded_chapters
        self.chapter_ids = chapter_ids
    
    def query(self, query: str) -> list[tuple[int, float]]:
        '''
        query: Query of interest.

        Search for the relevant Bible chapters to the query.
        '''
        if query.strip() == '' or len(self.encoded_chapters) == 0:
            return []
        
        encoded_query = self.bi_encoder_model.encode(query)
        scores = util.dot_score(encoded_query, self.encoded_chapters)[0].cpu().tolist()

        chapter_results = {}
        for chapter_id, result in tqdm(zip(self.chapter_ids, scores)):
            if chapter_id not in chapter_results:
                chapter_results[chapter_id] = []
            chapter_results[chapter_id].append(result)
        for chapter_id in chapter_results.keys():
            chapter_results[chapter_id] = (chapter_id, np.mean(chapter_results[chapter_id]).item())

        results = [result for result in chapter_results.values() if result[1] != 0]
        results = sorted(results, key=lambda chapter_score: chapter_score[1], reverse=True)

        return results
    
    def ranker_name(self) -> str:
        return 'bi_encoder_ranker'


class CrossEncoderRanker:
    """
    Cross-encoder ranker.
    """
    def __init__(self, cross_encoder_model_name: str, verses: dict[int, list[str]]) -> None:
        '''
        cross_encoder_model_name: Cross-encoder model.
        verses: Bible chapters and their verses.

        Search for the relevant Bible chapters to the query.
        '''
        self.cross_encoder_model = CrossEncoder(cross_encoder_model_name, max_length=512)
        self.verses = verses
    
    def score(self, chapterid: int, query: str) -> float:
        '''
        chapterid: Bible chapter id.
        query: Query of interest.

        Scores the relevance of the Bible chapter to the query.
        '''
        if query.strip() == '' or chapterid not in self.verses:
            return 0
        score = []
        for i in range(0, len(self.verses[chapterid]), 5):
            verses = " ".join(self.verses[chapterid][i:i + 5])
            verse_score = self.cross_encoder_model.predict([(query, verses)])[0]
            score.append(verse_score)
        return np.mean(score)
    
    def ranker_name(self) -> str:
        return 'cross_encoder_ranker'


class ColbertRanker:
    """
    Colbert ranker.
    """
    def __init__(self, colbert_model_name: str, chapter_collection_name: str) -> None:
        """
        colbert_model_name: ColBERT model.
        chapter_collection_name: Chapter collection name.

        Initializes ColBERT ranker.
        """
        self.colbert_model = LateInteractionTextEmbedding(colbert_model_name)
        self.colbert_client = QdrantClient(
            url="https://aabdb742-af72-430a-b07d-ac0fb2cdf7d8.us-east4-0.gcp.cloud.qdrant.io:6333",
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.hKeuZd7ey16S46OHrJmnJM9ocxgCSBheEbASQe5tchg"
        )
        self.chapter_collection_name = chapter_collection_name
        self.query_term = ""
        self.chapter_scores = {}

    def query(self, query: str) -> list[tuple[int, float]]:
        if query.strip() == "":
            return []

        self.query_term = query
        results = self.colbert_client.query_points(collection_name=self.chapter_collection_name,
                                                   query=list(self.colbert_model.query_embed(query))[0],
                                                   limit=1189).points
        results = [(result.id, result.score) for result in results if result.score != 0]
        results = sorted(results, key=lambda chapter_score: chapter_score[1], reverse=True)
        for chapterid, score in results:
            self.chapter_scores[chapterid]  = score

        return results

    def score(self, chapterid: int, query: str) -> float:
        if query != self.query_term:
            self.query(query)

        if chapterid not in self.chapter_scores:
            return 0
        return self.chapter_scores[chapterid]

    def ranker_name(self) -> str:
        return 'colbert_ranker'
