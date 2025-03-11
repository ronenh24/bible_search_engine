# Author: Ronen H

from bible_search_engine.indexing import BibleChapterIndex
from bible_search_engine.document_preprocessor import NLPTokenizer
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
from collections import Counter
from tqdm import tqdm


class TraditionalRanker():
    def __init__(self, bible_chapter_index: BibleChapterIndex, nlp_tokenizer: NLPTokenizer):
        '''
        Initializes the traditional ranker.

        bible_chapter_index: Bible chapter index.
        nlp_tokenizer: NLP tokenizer.
        '''
        pass

    def query(self, query: str) -> list[tuple[int, int]]:
        '''
        Search for the relevant Bible chapters to the query.

        query: Query of interest.

        Returns the most relevant Bible chapters for the query by highest score.
        '''
        pass

    def score(self, chapterid: int, query_parts: list[str]) -> int:
        '''
        Scores the relevance of the Bible chapter to the query.

        chapterid: Bible chapter id.
        query_parts: Tokenized query.

        Returns the score.
        '''
        pass

    def ranker_name(self) -> str:
        pass


# Shortest length ranking of relevant Bible chapters.
class RandomRanker(TraditionalRanker):
    def __init__(self, bible_chapter_index: BibleChapterIndex, nlp_tokenizer: NLPTokenizer) -> None:
        self.bible_chapter_index = bible_chapter_index
        self.nlp_tokenizer = nlp_tokenizer

    def query(self, query: str) -> list[tuple[int, int]]:
        results = []
        
        query_parts = self.nlp_tokenizer.tokenize(query)

        for chapterid in tqdm(self.bible_chapter_index.get_chapter_ids()):
            score = self.score(chapterid, query_parts)
            if score != 0:
                results.append((chapterid, score))

        results = sorted(results, key=lambda chapter_score: chapter_score[1], reverse=True)

        return results
    
    # Inverse Bible chapter length.
    def score(self, chapterid: int, query_parts: list[str]) -> int:
        score = 0

        set_query_parts = set([query_part for query_part in query_parts if query_part])
        chapter_vocab = self.bible_chapter_index.get_chapter_vocab(chapterid)

        if len(set_query_parts.intersection(chapter_vocab)) > 0:
            score = 1 / self.bible_chapter_index.get_chapter_metadata(chapterid)['chapter_length']
        
        return score
    
    def ranker_name(self) -> str:
        return 'random_ranker'


# TF-IDF ranking of relevant Bible chapters.
class TFIDFRanker(TraditionalRanker):
    def __init__(self, bible_chapter_index: BibleChapterIndex, nlp_tokenizer: NLPTokenizer) -> None:
        self.bible_chapter_index = bible_chapter_index
        self.nlp_tokenizer = nlp_tokenizer
    
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
    
    # TF-IDF.
    def score(self, chapterid: int, chapter_term_counts: dict[str, int], query_parts: list[str]) -> float:
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


# BM25 ranking of relevant Bible chapters.
class BM25Ranker(TraditionalRanker):
    def __init__(self, bible_chapter_index: BibleChapterIndex, nlp_tokenizer: NLPTokenizer,
                 b: float = 0.75, k1: float = 1.2, k3: float = 8) -> None:
        self.bible_chapter_index = bible_chapter_index
        self.nlp_tokenizer = nlp_tokenizer
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
    
    # BM25.
    def score(self, chapterid: int, chapter_term_counts: dict[str, int], query_parts: list[str]) -> float:
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


# Dirichlet LM ranking of relevant Bible chapters.
class DirichletLMRanker(TraditionalRanker):
    def __init__(self, bible_chapter_index: BibleChapterIndex, nlp_tokenizer: NLPTokenizer,
                 mu: float = 2000) -> None:
        self.bible_chapter_index = bible_chapter_index
        self.nlp_tokenizer = nlp_tokenizer
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
    
    # Dirichlet LM
    def score(self, chapterid: int, chapter_term_counts: dict[str, int], query_parts: list[str]) -> float:
        bible_statistics = self.bible_chapter_index.get_statistics()
        chapter_statistics = self.bible_chapter_index.get_chapter_metadata(chapterid)

        number_of_chapters = bible_statistics['number_of_chapters']
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
    def __init__(self, bi_encoder_model_name: str, encoded_chapters: np.ndarray, chapter_ids: list[int]) -> None:
        '''
        Initializes the bi-encoder ranker.

        bi_encoder_model_name: Bi-encoder model.
        encoded_chapters: Encoded Bible chapters from the bi-encoder model.
        chapter_ids: Bible chapter ids of the encoded Bible chapters.
        '''
        self.bi_encoder_model = SentenceTransformer(bi_encoder_model_name, device='cpu')
        self.encoded_chapters = encoded_chapters
        self.chapter_ids = chapter_ids
        self.chapterid_encoded_chapter = {}
        for chapterid, encoded_chapter in zip(chapter_ids, encoded_chapters):
            self.chapterid_encoded_chapter[chapterid] = encoded_chapter
    
    def query(self, query: str) -> list[tuple[int, float]]:
        '''
        Search for the relevant Bible chapters to the query.

        query: Query of interest.

        Returns the most relevant Bible chapters for the query by highest score.
        '''
        if query.strip() == '' or len(self.encoded_chapters) == 0:
            return []
        
        encoded_query = self.bi_encoder_model.encode(query)
        scores = util.dot_score(encoded_query, self.encoded_chapters)[0].cpu().tolist()

        results = [result for result in list(zip(self.chapter_ids, scores)) if result[1] != 0]
        results = sorted(results, key=lambda chapter_score: chapter_score[1], reverse=True)

        return results
    
    def score(self, chapterid: int, query: str) -> float:
        '''
        Scores the relevance of the Bible chapter to the query.

        chapterid: Bible chapter id.
        query: Query of interest.

        Returns the score.
        '''
        if query.strip() == '' or chapterid not in self.chapterid_encoded_chapter:
            return 0
        encoded_query = self.bi_encoder_model.encode(query)
        score = util.dot_score(encoded_query, self.chapterid_encoded_chapter[chapterid])[0].cpu().numpy()[0]
        return score
    
    def ranker_name(self) -> str:
        return 'bi_encoder_ranker'


class CrossEncoderRanker:
    def __init__(self, cross_encoder_model_name: str, verses: dict[int, list[str]]) -> None:
        '''
        Initializes the cross-encoder ranker.

        cross_encoder_model_name: Cross-encoder model.
        verses: Bible chapters and their verses.
        '''
        self.cross_encoder_model = CrossEncoder(cross_encoder_model_name, max_length=512)
        self.verses = verses
    
    def query(self, query: str) -> list[tuple[int, float]]:
        '''
        Search for the relevant Bible chapters to the query.

        query: Query of interest.

        Returns the most relevant Bible chapters for the query by highest score.
        '''
        if query.strip() == '':
            return []

        results = []
        for chapterid in tqdm(self.verses.keys()):
            score = self.score(chapterid, query)
            if score != 0:
                results.append((chapterid, score))

        results = sorted(results, key=lambda chapter_score: chapter_score[1], reverse=True)

        return results
    
    def score(self, chapterid: int, query: str) -> float:
        '''
        Scores the relevance of the Bible chapter to the query.

        chapterid: Bible chapter id.
        query: Query of interest.

        Returns the maximum Bible chapter verse score.
        '''
        if query.strip() == '' or chapterid not in self.verses:
            return 0
        score = None
        for i in range(0, len(self.verses[chapterid]), 5):
            verses = " ".join(self.verses[chapterid][i:i + 5])
            verse_score = self.cross_encoder_model.predict([(query, verses)])[0]
            if score is None:
                score = verse_score
            score = max(score, verse_score)
        return score
    
    def ranker_name(self) -> str:
        return 'cross_encoder_ranker'


#if __name__ == '__main__':
    #import os
    #import orjson

    #if not os.path.isfile('bible_data/encoded_chapters.npy'):
        #bi_encoder_model = SentenceTransformer('msmarco-distilbert-dot-v5', device='cpu')
        #chapter_ids = []
        #chapter_texts = []
        #with open('bible_data/old_testament.jsonl', 'rb') as old_testament_file:
            #for bible_chapter_line in tqdm(old_testament_file):
                #bible_chapter = orjson.loads(bible_chapter_line)
                #chapter_ids.append(bible_chapter['chapterid'])
                #chapter_texts.append(' '.join(bible_chapter['verses'].values()))
        #with open('bible_data/new_testament.jsonl', 'rb') as new_testament_file:
            #for bible_chapter_line in tqdm(new_testament_file):
                #bible_chapter = orjson.loads(bible_chapter_line)
                #chapter_ids.append(bible_chapter['chapterid'])
                #chapter_texts.append(' '.join(bible_chapter['verses'].values()))
        #encoded_chapters = bi_encoder_model.encode(chapter_texts, show_progress_bar=True)
        #np.save('bible_data/encoded_chapters.npy', encoded_chapters)
        #with open('bible_data/chapterids.json', 'xb') as chapterids_file:
            #chapterids_file.write(orjson.dumps(chapter_ids))

