"""
Author: Ronen Huang
"""

from collections import Counter
from bisect import insort_left, bisect_left
import os
from importlib.resources.abc import Traversable
from importlib.resources import files
from tqdm import tqdm
import orjson
from bible_search_engine.components.preprocess import NLPTokenizer


class BibleChapterIndex:
    """
    Inverted Bible chapter index.
    """
    def __init__(self):
        '''
        Initializes the Bible chapter index.
        '''
        self.index = {}
        self.statistics = {}
        self.statistics['number_of_chapters'] = 0
        self.statistics['total_token_count'] = 0
        self.statistics['mean_chapter_length'] = 0
        self.statistics['number_of_verses'] = 0
        self.statistics['mean_verse_length'] = 0
        self.statistics['unique_token_count'] = 0
        self.chapter_metadata = {}
        self.term_metadata = {}
        self.chapter_vocab = {}

    def add_chapter(self, chapterid: int, num_verses: int,
                    tokens: list[str]) -> None:
        '''
        Index the Bible chapter.

        chapterid: Bible chapter id.
        num_verses: Number of verses in Bible chapter.
        tokens: Tokenized Bible chapter.
        '''
        # Update number of Bible chapters.
        self.statistics['number_of_chapters'] += 1

        # Update length of all Bible chapters.
        self.statistics['total_token_count'] += len(tokens)

        # Update number of verses.
        self.statistics['number_of_verses'] += num_verses

        token_counts = Counter(
            [token for token in tokens if token is not None]
        )

        # Store Bible chapter length, number of relevant tokens,
        # and number of verses.
        self.chapter_metadata[chapterid] = {
            'chapter_length': len(tokens),
            'number_of_relevant_terms': len(token_counts.keys()),
            'number_of_verses': num_verses
        }

        # Store Bible chapter unique relevant tokens.
        self.chapter_vocab[chapterid] = set(token_counts.keys())

        for indexed_token, count in token_counts.items():
            # Add relevant token to index.
            if indexed_token not in self.index:
                self.index[indexed_token] = []

                # Update number of unique relevant tokens.
                self.statistics['unique_token_count'] += 1

                self.term_metadata[indexed_token] = {
                    'number_of_chapters_appears': 0,
                    'total_term_frequency': 0
                }
            # Add Bible chapter to relevant token postings.
            insort_left(
                self.index[indexed_token], (chapterid, count),
                key=lambda chapter_count: chapter_count[0]
            )

            # Update number of Bible chapters relevant token appears.
            self.term_metadata[indexed_token]['number_of_chapters_appears'] +=\
                1

            # Update number of times relevant token appears.
            self.term_metadata[indexed_token]['total_term_frequency'] += count

        # Update average Bible chapter length.
        self.statistics['mean_chapter_length'] =\
            self.statistics['total_token_count'] /\
            self.statistics['number_of_chapters']

        # Update average verse length.
        self.statistics['mean_verse_length'] =\
            self.statistics['total_token_count'] /\
            self.statistics['number_of_verses']

    def remove_chapter(self, chapterid: int) -> None:
        '''
        Remove Bible chapter from index.

        chapterid: Bible chapter id.
        '''
        # Update length of all Bible chapters.
        self.statistics['total_token_count'] -=\
            self.chapter_metadata[chapterid]['chapter_length']

        # Update number of Bible chapters.
        self.statistics['number_of_chapters'] -= 1

        # Update number of verses.
        self.statistics['number_of_verses'] -=\
            self.chapter_metadata[chapterid]['number_of_verses']

        for token in self.chapter_vocab[chapterid]:
            # Update number of Bible chapters relevant token appears.
            self.term_metadata[token]['number_of_chapters_appears'] -= 1

            # Remove Bible chapter from relevant token postings.
            chapter_token_count = self.index[token].pop(
                bisect_left(
                    self.index[token], chapterid,
                    key=lambda chapter_count: chapter_count[0]
                    )
            )[1]

            # Update number of times relevant token appears.
            self.term_metadata[token]['total_term_frequency'] -=\
                chapter_token_count

            # Relevant token no longer appears.
            if self.term_metadata[token]['number_of_chapters_appears'] == 0:
                # Update number of unique relevant tokens.
                self.statistics['unique_token_count'] -= 1
                del self.index[token]
                del self.term_metadata[token]

        del self.chapter_metadata[chapterid]
        del self.chapter_vocab[chapterid]

        # Update average Bible chapter length and average verse length.
        if self.statistics['number_of_chapters'] == 0:
            self.statistics['mean_chapter_length'] = 0
            self.statistics['mean_verse_length'] = 0
        else:
            self.statistics['mean_chapter_length'] =\
                self.statistics['total_token_count'] /\
                self.statistics['number_of_chapters']
        if self.statistics['number_of_verses'] == 0:
            self.statistics['mean_verse_length'] = 0
        else:
            self.statistics['mean_verse_length'] =\
                self.statistics['total_token_count'] /\
                self.statistics['number_of_verses']

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        '''
        Obtains Bible chapters with the term along with the frequencies.

        term: Term of interest.

        Returns Bible chapters with the term along with the frequencies.
        '''
        if term not in self.index:
            return []
        else:
            return self.index[term]

    def get_chapter_metadata(self, chapterid: int) -> dict[str, int]:
        '''
        Obtains Bible chapter metadata.

        chapterid: Bible chapter id.

        Returns Bible chapter metadata.
        '''
        if chapterid not in self.chapter_metadata:
            return {}
        else:
            return self.chapter_metadata[chapterid]

    def get_term_metadata(self, term: str) -> dict[str, int]:
        '''
        Obtains term metadata.

        term: Term of interest.

        Returns term metadata.
        '''
        if term not in self.index:
            return {}
        else:
            return self.term_metadata[term]

    def get_statistics(self) -> dict[str, int]:
        '''
        Obtains index statistics.

        Returns index statistics.
        '''
        return self.statistics

    def get_chapter_vocab(self, chapterid: int) -> set[str]:
        '''
        Obtain the relevant vocabulary of the Bible chapter.

        chapterid: Bible chapter id.

        Returns relevant vocabulary of Bible chapter.
        '''
        if chapterid not in self.chapter_vocab:
            return set()
        else:
            return self.chapter_vocab[chapterid]

    def get_chapter_ids(self):
        '''
        Obtains the Bible chapter ids in the index.

        Returns the Bible chapter ids in the index.
        '''
        return self.chapter_metadata.keys()

    def get_chapter_term_freq(self, chapterid: int, term: str) -> int:
        '''
        Obtain the number of times the term appeared in the Bible chapter.

        chapterid: Bible chapter id.
        term: Term of interest.

        Returns the number of times the term appeared in the Bible chapter.
        '''
        if term not in self.index:
            return 0
        else:
            term_postings = self.index[term]
            term_postings_chapter = term_postings[
                bisect_left(
                    term_postings, chapterid,
                    key=lambda chapter_count: chapter_count[0]
                )
            ]
            # Term does not appear in Bible chapter.
            if term_postings_chapter[0] != chapterid:
                return 0
            else:
                return term_postings_chapter[1]

    def save(self, bible_index_dir: str) -> None:
        '''
        Save the index.

        bible_index_dir: Directory to save index to.
        '''
        if not os.path.isdir(bible_index_dir):
            os.mkdir(bible_index_dir)
        if not os.path.isfile(bible_index_dir + '/__init__.py'):
            init_file = open(
                bible_index_dir + '/__init__.py', 'x',
                encoding='utf-8'
            )
            init_file.close()
        self._save_index(bible_index_dir)
        self._save_stats(bible_index_dir)
        self._save_chapter_metadata(bible_index_dir)
        self._save_term_metadata(bible_index_dir)
        self._save_chapter_vocab(bible_index_dir)
        print('Saved Bible Index to ' + bible_index_dir)

    def _save_index(self, bible_index_dir: str) -> None:
        index_file_name = bible_index_dir + '/index.jsonl'
        with open(index_file_name, "wb") as index_file:
            for term, postings in tqdm(self.index.items()):
                index_file.write(
                    orjson.dumps(
                        {
                            'term': term,
                            'postings': postings
                        }
                    ) + b'\n'
                )

    def _save_stats(self, bible_index_dir: str) -> None:
        stats_file_name = bible_index_dir + '/statistics.json'
        with open(stats_file_name, 'wb') as stats_file:
            stats_file.write(orjson.dumps(self.statistics))

    def _save_chapter_metadata(self, bible_index_dir: str) -> None:
        chapter_metadata_file_name =\
            bible_index_dir + '/chapter_metadata.jsonl'
        with open(chapter_metadata_file_name, 'wb') as chapter_metadata_file:
            for chapterid, metadata in tqdm(self.chapter_metadata.items()):
                chapter_metadata_file.write(
                    orjson.dumps(
                        {
                            'chapterid': chapterid,
                            'chapter_metadata': metadata
                        }
                    ) + b'\n'
                )

    def _save_term_metadata(self, bible_index_dir: str) -> None:
        term_metadata_file_name = bible_index_dir + '/term_metadata.jsonl'
        with open(term_metadata_file_name, 'wb') as term_metadata_file:
            for term, metadata in tqdm(self.term_metadata.items()):
                term_metadata_file.write(
                    orjson.dumps(
                        {
                            'term': term,
                            'term_metadata': metadata
                        }
                    ) + b'\n'
                )

    def _save_chapter_vocab(self, bible_index_dir: str) -> None:
        chapter_vocab_file_name = bible_index_dir + '/chapter_vocab.jsonl'
        with open(chapter_vocab_file_name, 'wb') as chapter_vocab_file:
            for chapterid, vocab in tqdm(self.chapter_vocab.items()):
                chapter_vocab_file.write(
                    orjson.dumps(
                        {
                            'chapterid': chapterid,
                            'chapter_vocab': list(vocab)
                        }
                    ) + b'\n'
                )

    def load(self, bible_index_dir: str | Traversable) -> None:
        '''
        Loads the index.

        bible_index_dir: Directory to load index from.
        '''
        if isinstance(bible_index_dir, str):
            bible_index_dir = files(bible_index_dir.replace("/", "."))
        self.index.clear()
        self.statistics.clear()
        self.chapter_metadata.clear()
        self.term_metadata.clear()
        self.chapter_vocab.clear()
        if not bible_index_dir.is_dir:
            raise Exception('Index directory does not exist.')
        self._load_index(bible_index_dir)
        self._load_stats(bible_index_dir)
        self._load_chapter_metadata(bible_index_dir)
        self._load_term_metadata(bible_index_dir)
        self._load_chapter_vocab(bible_index_dir)
        print('Loaded Bible Index from ' + str(bible_index_dir))

    def _load_index(self, bible_index_dir: Traversable) -> None:
        index_path = bible_index_dir / "index.jsonl"
        if not index_path.is_file:
            raise Exception('Index file does not exist.')
        with index_path.open('rb') as index_file:
            for term_postings_line in tqdm(index_file):
                term_postings = orjson.loads(term_postings_line)
                self.index[term_postings['term']] =\
                    [
                        tuple(chapter_freq)
                        for chapter_freq in term_postings['postings']
                    ]

    def _load_stats(self, bible_index_dir: Traversable) -> None:
        stats_path = bible_index_dir / "statistics.json"
        if not stats_path.is_file:
            raise Exception('Statistics file does not exist.')
        with stats_path.open("rb") as stats_file:
            self.statistics = orjson.loads(stats_file.readline())

    def _load_chapter_metadata(self, bible_index_dir: Traversable) ->\
            None:
        chapter_metadata_path = bible_index_dir / "chapter_metadata.jsonl"
        if not chapter_metadata_path.is_file:
            raise Exception('Chapter metadata file does not exist.')
        with chapter_metadata_path.open("rb") as chapter_metadata_file:
            for chapter_metadata_line in tqdm(chapter_metadata_file):
                chapter_metadata = orjson.loads(chapter_metadata_line)
                self.chapter_metadata[chapter_metadata['chapterid']] =\
                    chapter_metadata['chapter_metadata']

    def _load_term_metadata(self, bible_index_dir: str | Traversable) -> None:
        term_metadata_path = bible_index_dir / "term_metadata.jsonl"
        if not term_metadata_path.is_file:
            raise Exception('Term metadata file does not exist.')
        with term_metadata_path.open("rb") as term_metadata_file:
            for term_metadata_line in tqdm(term_metadata_file):
                term_metadata = orjson.loads(term_metadata_line)
                self.term_metadata[term_metadata['term']] =\
                    term_metadata['term_metadata']

    def _load_chapter_vocab(self, bible_index_dir: str | Traversable) -> None:
        chapter_vocab_path = bible_index_dir / "chapter_vocab.jsonl"
        if not chapter_vocab_path.is_file:
            raise Exception('Chapter vocabulary file does not exist.')
        with chapter_vocab_path.open("rb") as chapter_vocab_file:
            for chapter_vocab_line in tqdm(chapter_vocab_file):
                chapter_vocab = orjson.loads(chapter_vocab_line)
                self.chapter_vocab[chapter_vocab['chapterid']] =\
                    set(chapter_vocab['chapter_vocab'])


def create_bible_index(old_testament_path: str, new_testament_path: str,
                       nlp_tokenizer: NLPTokenizer) -> BibleChapterIndex:
    '''
    old_testament_path: Path to Old Testament Bible chapters.
    new_testament_path: Path to New Testament Bible chapters.
    nlp_tokenizer: NLP tokenizer to tokenize Bible chapters.

    Creates the Bible index.
    '''
    bible_chapter_index = BibleChapterIndex()

    # Index Old Testament Bible chapters.
    if not os.path.isfile(old_testament_path):
        raise Exception('Old Testament file does not exist.')
    with open(old_testament_path, 'rb') as old_testament_file:
        for bible_chapter_line in tqdm(old_testament_file):
            bible_chapter = orjson.loads(bible_chapter_line)
            bible_chapter_id = bible_chapter['chapterid']
            bible_chapter_text = bible_chapter['verses'].values()
            bible_chapter_index.add_chapter(
                bible_chapter_id, bible_chapter['num_verses'],
                nlp_tokenizer.tokenize(bible_chapter_text, False)
            )

    # Index New Testament Bible chapters.
    if not os.path.isfile(new_testament_path):
        raise Exception('New Testament file does not exist.')
    with open(new_testament_path, 'rb') as new_testament_file:
        for bible_chapter_line in tqdm(new_testament_file):
            bible_chapter = orjson.loads(bible_chapter_line)
            bible_chapter_id = bible_chapter['chapterid']
            bible_chapter_text = bible_chapter['verses'].values()
            bible_chapter_index.add_chapter(
                bible_chapter_id, bible_chapter['num_verses'],
                nlp_tokenizer.tokenize(bible_chapter_text, False)
            )

    return bible_chapter_index
