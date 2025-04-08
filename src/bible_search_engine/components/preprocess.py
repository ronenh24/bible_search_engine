# Author: Ronen Huang

import spacy
from transformers import pipeline


class NLPTokenizer:
    """
    Natural Language Processing tokenizer for Bible chapters.
    """
    def __init__(self, lowercase: bool = True, stopword: bool = True, multiword_expr: bool = False,
                 punctuation: bool = False, lemmatize: bool = True) -> None:
        '''
        Initializes the NLP tokenizer for the Bible chapters.

        lowercase: True to lowercase tokens. Defaults to True.
        stopword: True to filter stopwords. Defaults to True.
        multiword_expr: True to recognize multiword expressions. Defaults to False.
        punctuation: True to include punctuation. Defaults to False.
        lemmatize: True to apply lemmatization. Defaults to True.
        '''
        self.spacy_tokenizer = spacy.load('en_core_web_lg')
        if multiword_expr:
            self.spacy_tokenizer.add_pipe('merge_entities')
        self.lowercase = lowercase
        self.stopword = stopword
        self.punctuation = punctuation
        self.lemmatize = lemmatize
        self.flan_expansion = pipeline("text2text-generation", model="google/flan-t5-large")
    
    def tokenize(self, chapter_text: list[str] | str) -> list[str]:
        '''
        chapter_text: The Bible chapter text.

        Tokenizes the Bible chapter text with NLP.
        '''
        if isinstance(chapter_text, str):
            chapter_text = self.get_keywords(chapter_text)
        else:
            chapter_text = " ".join([self.get_keywords(verse_text) for verse_text in chapter_text])

        tokenized_chapter_text = self.spacy_tokenizer(chapter_text)

        if not self.stopword: # No stopwords included.
            tokenized_chapter_text = [None if token.is_stop else token for token in tokenized_chapter_text]
        
        if not self.punctuation: # No punctuation included.
            tokenized_chapter_text = [None if token is None or token.is_punct else token
                                      for token in tokenized_chapter_text]

        if self.lemmatize: # Lemmatization applied.
            tokens = [token.lemma_ if token else None for token in tokenized_chapter_text]
        else:
            tokens = [token.text if token else None for token in tokenized_chapter_text]
        
        if self.lowercase: # Lowercasing applied.
            tokens = [token.lower() if token else None for token in tokens]
        
        return tokens

    def get_keywords(self, verse_text: str) -> str:
        """
        verse_text: The Bible verse text.

        Appends keywords to the end.
        """
        prompt = "Determine keywords for text: " + verse_text
        return verse_text + " " + " ".join(self.flan_expansion(prompt)[0]["generated_text"].split(", "))
