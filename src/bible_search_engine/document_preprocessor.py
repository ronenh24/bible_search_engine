# Author: Ronen H

import spacy


class NLPTokenizer:
    def __init__(self, lowercase: bool = True, stopword: bool = False,
                 multiword_expr: bool = False, punctuation: bool = False,
                 lemmatize: bool = True) -> None:
        '''
        Initializes the NLP tokenizer for the Bible chapters.

        lowercase: True to lowercase tokens. Defaults to True.
        stopword: True to include stopwords. Defaults to False.
        multiword_expr: True to recognize multiword expressions. Defaults to False.
        punctuation: True to include punctuation. Defaults to False.
        lemmatize: True to apply lemmatization. Defaults to True.
        '''
        self.spacy_tokenizer = spacy.load('en_core_web_sm')
        if multiword_expr:
            self.spacy_tokenizer.add_pipe('merge_entities')
        self.lowercase = lowercase
        self.stopword = stopword
        self.punctuation = punctuation
        self.lemmatize = lemmatize
    
    def tokenize(self, chapter_text: str) -> list[str]:
        '''
        Tokenizes the Bible chapter text with NLP.

        chapter_text: The Bible chapter text.

        Returns list of tokens.
        '''
        tokenized_chapter_text = self.spacy_tokenizer(chapter_text)

        if not self.stopword: # No stopwords included.
            tokenized_chapter_text = [None if token.is_stop else token for token in tokenized_chapter_text]
        
        if not self.punctuation: # No punctuation included.
            tokenized_chapter_text = [None if token is None or token.is_punct else token
                                      for token in tokenized_chapter_text]
        
        tokens = []

        if self.lemmatize: # Lemmatization applied.
            tokens = [token.lemma_ if token else None for token in tokenized_chapter_text]
        else:
            tokens = [token.text if token else None for token in tokenized_chapter_text]
        
        if self.lowercase: # Lowercasing applied.
            tokens = [token.lower() if token else None for token in tokens]
        
        return tokens

