"""
Author: Ronen Huang
"""

import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from nltk.corpus.reader.wordnet import Synset
from transformers import pipeline


class NLPTokenizer:
    """
    Natural Language Processing tokenizer for Bible chapters.
    """
    def __init__(self, lowercase: bool = True, stopword: bool = False,
                 multiword_expr: bool = False, punctuation: bool = False,
                 lemmatize: bool = True) -> None:
        '''
        Initializes the NLP tokenizer for the Bible chapters.

        lowercase: True to lowercase tokens. Defaults to True.
        stopword: True to include stopwords. Defaults to False.
        multiword_expr: True to recognize multiword expressions.
                        Defaults to False.
        punctuation: True to include punctuation. Defaults to False.
        lemmatize: True to apply lemmatization. Defaults to True.
        '''
        self.spacy_tokenizer = spacy.load('en_core_web_lg')
        self.spacy_tokenizer.add_pipe("spacy_wordnet", after='tagger')
        if multiword_expr:
            self.spacy_tokenizer.add_pipe('merge_entities')
        self.lowercase = lowercase
        self.stopword = stopword
        self.punctuation = punctuation
        self.lemmatize = lemmatize
        self.flan_expansion = pipeline(
            "text2text-generation", model="google/flan-t5-large"
        )

    def tokenize(self, chapter_text: list[str] | str,
                 query: bool = True) -> list[str]:
        '''
        chapter_text: The Bible chapter text.

        Tokenizes the Bible chapter text with NLP.
        '''
        if isinstance(chapter_text, str):
            chapter_text = self.get_keywords(chapter_text)
        else:
            chapter_text = " ".join(
                [
                    self.get_keywords(verse_text)
                    for verse_text in chapter_text
                ]
            )

        tokens = []

        for token in self.spacy_tokenizer(chapter_text):
            if not self.stopword and token.is_stop:
                continue

            if not self.punctuation and token.is_punct:
                continue

            token_list = []
            if self.lemmatize:
                synsets = token._.wordnet.synsets()
                if query and synsets:
                    token_list = self.get_lemmas(synsets)
                else:
                    token_list.append(token.lemma_)
            else:
                token_list.append(token.text)
            
            if self.lowercase:
                for i, t in enumerate(token_list):
                    token_list[i] = t.lower()
            
            tokens.extend(token_list)

        return tokens

    def get_keywords(self, verse_text: str) -> str:
        """
        verse_text: The Bible verse text.

        Appends keywords to the end.
        """
        prompt = "Determine keywords for text: " + verse_text
        return verse_text + " " + " ".join(
            self.flan_expansion(prompt)[0]["generated_text"].split(", ")
        )

    def get_lemmas(self, synsets: list[Synset]) -> list[str]:
        """
        Get unique lemmas from sysnets.
        """
        lemmas = []
        for syn in synsets:
            for lemma in syn.lemma_names():

                for lemma_part in self.spacy_tokenizer(
                    lemma.replace("_", " ")
                ):
                    if not self.stopword and lemma_part.is_stop:
                        continue

                    if not self.punctuation and lemma_part.is_punct:
                        continue

                    if self.lemmatize:
                        lemmas.append(lemma_part.lemma_)
                    else:
                        lemmas.append(lemma_part.text)

        return list(set(lemmas))
