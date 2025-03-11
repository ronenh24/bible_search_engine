# Author
Ronen H  

# Time Frame
October 2023 to December 2023, March 2025 (deploy as package)

# Data Source
The text files for each book of the King James Bible can be accessed via [https://archive.org/details/kjv-text-files](https://archive.org/details/kjv-text-files).

# Install Bible Search Engine
The Bible Search Engine can be downloaded by pip.
```commandline
pip install bible_search_engine
```


# How to Run Bible Search Engine
The Bible search engine can be created with the `create_bible_search_engine` function from `pipeline.py`.  

For example,
```
from bible_search_engine.pipeline import create_bible_search_engine
bible_search_engine = create_bible_search_engine()
```
.

It is represented by the `BibleSearchEngine` class which has a `search` function that returns a list of ranked results for the given query. Each result is in dictionary form as
```
{'chapterid': chapter id, 'score': chapter score, 'chapter': chapter title, 'verses': chapter verses}
```
.  

For example, to search for the query "How does God show his mercy?"
```
bible_search_engine.search('How does God show his mercy?')
```
yields
```
[{'chapterid': 614, 'score': 78.20862579345703, 'chapter': 'Psalms 136', 'verses': 'verses': {'1': 'O give thanks unto the LORD; for he is good: for his mercy endureth for ever.', '2': 'O give thanks unto the God of gods: for his mercy endureth for ever.', ...}}, {'chapterid': 717, 'score': 75.48340606689453, 'chapter': 'Isaiah 38', 'verses': {'1': 'In those days was Hezekiah sick unto death. And Isaiah the prophet the son of Amoz came unto him, and said unto him, Thus saith the LORD, Set thine house in order: for thou shalt die, and not live.', '2': 'Then Hezekiah turned his face toward the wall, and prayed unto the LORD,', ...}}, ...]
```
.  

# File Information
- The **bible_basic_statistics** directory contains the Bible chapter statistics **chapter_statistics.csv** and the train and test query relevance score distribution **train_rel_scores_dist.jpg** and **test_rel_scores_dist.jpg** respectively. The process to obtain them is in `basic_bible_statistics.ipynb`.
- The **bible_data** directory contains the Old Testament chapters data **old_testament.jsonl**, the New Testament chapters data **new_testament.jsonl**, the encoded bible chapters by the "msmarco-distilbert-dot-v5" bi-encoder **encoded_chapters.npy**, and the chapter ids in order of the encoded bible chapters **chapterids.json**.
- The **bible_index** directory is the stored Bible chapter index.
- The **bible_queries_relevances** directories contains the train queries relevance scores data **train_queries_relevances.csv** and the test queries relevance scores data **test_queries_relevances.csv**. The process to create them is in `query_results.ipynb`.
- The **kjv-text-files** directory contains the text files of each Bible book.
- The **rankers** directory contains the performances of the baselines and Bible search engine on the test data **results.csv** and the bar plot of it **results.jpg**. The process to evaluate them is in `evaluate_rankers.ipynb`.
- The `get_bible_data` function from `preprocess_bible_data.py` preprocesses the text files of each Bible book to Old Testament chapters data and New Testament chapters data.
- The `NLPTokenizer` is in `document_preprocessor.py`.
- The `BibleChapterIndex` and the `create_bible_index` function is in `indexing.py`.
- The rankers `RandomRanker`, `TFIDFRanker`, `BM25Ranker`, `DirichletLMRanker`, `BiEncoderRanker`, `CrossEncoderRanker` in `rankers.py` have a `query` function and a `score` function.
- The `L2RRanker` in `l2r.py` has a `train` function and a `query` function and the `L2RFeatureExtractor` has a `get_features` function.
- The `Relevance` class in `relevance.py` has a `precision` function and `evaluate_ranker_results` function.

# Improvements
- Transition to New International Version
- Incorporate Large Language Models
- Bayesian Hyperparameter Optimization
- Search Speed
- Online Learning

# References
Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-strength Natural Language Processing in Python. doi:10.5281/zenodo.1212303  

Internet Archive. (2021). Retrieved from [https://archive.org/details/kjv-text-files](https://archive.org/details/kjv-text-files)  

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., . . . Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In *Proceedings of the 31st International Conference on Neural Information Processing Systems* (pp. 3149–3157). Curran Associates Inc.  

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics. Retrieved from [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)