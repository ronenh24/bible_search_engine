# Bible Chapter Search Engine

## Author
Ronen Huang  

## Time Frame
October 2023 to December 2023, March 2025 - Present (deploy as package)

## Data Source
The web pages for each book of the New International Version Bible can be accessed via [https://www.biblestudytools.com/](https://www.biblestudytools.com/).

## Install Bible Search Engine
The Bible Search Engine can be downloaded by pip.
```commandline
pip install bible_search_engine
```


## How to Run Bible Search Engine
The Bible search engine can be created with the `create_bible_search_engine` function from `pipeline.py`.  

For example,
```python
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
```python
bible_search_engine.search('How does God show his mercy?')
```
yields
```
[
    {'chapterid': 1164, 'score': 0.7548669934272766, 'chapter': '1 John 5',
        'verses': {
            '1': 'Everyone who believes that Jesus is the Christ is born of God, and everyone who loves the father loves his child as well.',
            '2': 'This is how we know that we love the children of God: by loving God and carrying out his commands.',
            ...
        }
    },
    {'chapterid': 1162, 'score': 0.7545916557312011, 'chapter': '1 John 3',
        'verses': {
            '1': 'See what great love the Father has lavished on us, that we should be called children of God! And that is what we are! The reason the world does not know us is that it did not know him.',
            '2': 'Dear friends, now we are children of God, and what we will be has not yet been made known. But we know that when Christ appears, we shall be like him, for we shall see him as he is.',
            ...
        }
    },
    ...
]
```
.  

## File Information
- The `get_bible_data` function from [`get_bible_data.py`](src/bible_search_engine/components/get_bible_data.py) obtains the web pages of each Bible chapter to Old Testament chapters data and New Testament chapters data.
  - Libraries:
    - [`requests`](https://requests.readthedocs.io/en/latest/) to retrieve HTML.
    - [`BeautifulSoup`](https://www.crummy.com/software/BeautifulSoup/) to retrieve text.
  - Usage Example
    ```python
    from bible_search_engine.components.get_bible_data import get_bible_data
    
    get_bible_data()
    ```
- The `NLPTokenizer` class is in [`preprocess.py`](src/bible_search_engine/components/preprocess.py). The `tokenize` function splits the Bible chapter text into tokens for indexing.
  - Libraries:
    - [`transformers`](https://huggingface.co/docs/transformers/en/index) with [`google/flan-t5-large`](https://huggingface.co/google/flan-t5-large) model to expand chapters and queries.
    - [`spacy`](https://spacy.io/) to tokenize text.
  - Usage Example
    ```python
    from bible_search_engine.components.preprocess import get_bible_data
    
    query = "Who is Jesus?"
    
    nlp_tokenizer = NLPTokenizer()
    query_tokens = nlp_tokenizer.tokenize(query)
    ```
- The `BibleChapterIndex` class is in [`indexing.py`](src/bible_search_engine/components/indexing.py). The `create_bible_index` function creates an inverted index by chapter.
  - Usage Example
    ```python
    from bible_search_engine.components.indexing import create_bible_index
    
    old_testament_path = 'bible_search_engine/bible_data/old_testament_niv.jsonl'
    new_testament_path = 'bible_search_engine/bible_data/new_testament_niv.jsonl'
    
    bible_chapter_index_path = 'bible_search_engine/bible_index_niv'
    bible_chapter_index = create_bible_index(old_testament_path, new_testament_path, nlp_tokenizer)
    bible_chapter_index.save()
    ```
- The rankers `RandomRanker`, `TFIDFRanker`, `BM25Ranker`, `DirichletLMRanker`, `BiEncoderRanker`, `ColbertRanker` in [`ranker.py`](src/bible_search_engine/components/ranker.py) have a `query` function and a `score` function.
  - Libraries:
    - [`sentence_transformers`](https://sbert.net/) with [`msmarco-distilbert-dot-v5`](https://huggingface.co/sentence-transformers/msmarco-distilbert-dot-v5) model to retrieve initial results with bi-encoder.
    - [`qdrant`](https://qdrant.tech/documentation/fastembed/fastembed-colbert/) with [`colbert-ir/colbertv2.0`](https://huggingface.co/colbert-ir/colbertv2.0) model to add ColBERT embedding similarity feature for learning to rank.
  - Usage Example
    ```python
    from bible_search_engine.components.ranker import (TFIDFRanker, BM25Ranker, DirichletLMRanker,
                                                       BiEncoderRanker, ColbertRanker)
    import orjson
    import numpy as np
    
    tf_idf_ranker = TFIDFRanker(bible_chapter_index, nlp_tokenizer)
    bm25_ranker = BM25Ranker(bible_chapter_index, nlp_tokenizer)
    dirichlet_lm_ranker = DirichletLMRanker(bible_chapter_index, nlp_tokenizer)
    
    colbert_ranker = ColbertRanker("colbert-ir/colbertv2.0", "bible_chapters")
    
    encoded_chapters_path = 'bible_search_engine/bible_data/encoded_chapters_niv.npy'
    chapter_ids_path = 'bible_search_engine/bible_data/chapterids_niv.json'
    encoded_chapters = np.load(encoded_chapters_path)
    with chapter_ids_path.open('rb') as chapter_ids_file:
        chapter_ids = orjson.loads(chapter_ids_file.readline())
    
    bi_encoder_ranker = BiEncoderRanker('msmarco-distilbert-dot-v5', encoded_chapters, chapter_ids)
    
    tf_idf_results = tf_idf_ranker.query(query)
    bm25_results = bm25_ranker.query(query)
    dirichlet_lm_results = dirichlet_lm_ranker.query(query)
    bi_encoder_results = bi_encoder_ranker.query(query)
    colbert_results = colbert_ranker.query(query)
    ```
- The `L2RRanker` class in [`l2r.py`](src/bible_search_engine/components/l2r.py) has a `train` function and a `query` function. The `L2RFeatureExtractor` has a `get_features` function.
  - Libraries:
    - [`lightgbm`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html) to implement learning to rank for reranking of top 100 initial results.
    - [`optuna`](https://optuna.readthedocs.io/en/stable/index.html) to tune hyperparameters `learning_rate`, `num_iterations`, `max_depth`, `num_leaves`, `min_data_in_leaf`.
  - Usage Example
    ```python
    from bible_search_engine.components.l2r import L2RRanker, L2RFeatureExtractor
    
    train_queries_path = 'bible_search_engine/bible_queries_relevances/train_queries_relevances.csv'
    test_queries_path = 'bible_search_engine/bible_queries_relevances/test_queries_relevances.csv'
    
    l2r_feature_extractor = L2RFeatureExtractor(bible_chapter_index, nlp_tokenizer, tf_idf_ranker,
                                                bm25_ranker, dirichlet_lm_ranker, colbert_ranker)
    l2r_ranker = L2RRanker(bible_chapter_index, nlp_tokenizer, bi_encoder_ranker, l2r_feature_extractor)
    l2r_ranker.train(train_queries_path, test_queries_path)
    
    l2r_results = l2r_ranker.query(query)
    ```
- The `Relevance` class in [`relevance.py`](src/bible_search_engine/components/relevance.py) has a `precision` function and `evaluate_ranker_results` function. The default is precision at 15 Bible chapters.
  - Usage Example
    ```python
    from bible_search_engine.components.relevance import Relevance
    
    test_eval = Relevance(test_queries_data)
    tf_idf_eval_results = test_eval.evaluate_ranker_results(tf_idf_ranker)
    bm25_eval_results = test_eval.evaluate_ranker_results(bm25_ranker)
    dirichlet_lm_eval_results = test_eval.evaluate_ranker_results(dirichlet_lm_ranker)
    bi_encoder_eval_results = test_eval.evaluate_ranker_results(bi_encoder_ranker)
    l2r_eval_results = test_eval.evaluate_ranker_results(l2r_ranker)
    ```
- The **[bible_basic_statistics](src/bible_basic_statistics)** directory contains the Bible chapter statistics **[chapter_statistics.csv](src/bible_basic_statistics/chapter_statistics.csv)** and the train and test query relevance score distribution **[train_rel_scores_dist.jpg](src/bible_basic_statistics/train_rel_scores_dist.jpg)** and **[test_rel_scores_dist.jpg](src/bible_basic_statistics/test_rel_scores_dist.jpg)** respectively. The process to obtain them is in [`basic_bible_statistics.ipynb`](src/basic_bible_statistics.ipynb).
- The **[bible_data](src/bible_search_engine/bible_data)** directory contains the Old Testament chapters data **[old_testament_niv.jsonl](src/bible_search_engine/bible_data/old_testament_niv.jsonl)**, the New Testament chapters data **[new_testament_niv.jsonl](src/bible_search_engine/bible_data/new_testament_niv.jsonl)**, the encoded bible chapters by the bi-encoder **[encoded_chapters_niv.npy](src/bible_search_engine/bible_data/encoded_chapters_niv.npy)**, and the chapter ids in order of the encoded bible chapters **[chapterids_niv.json](src/bible_search_engine/bible_data/chapterids_niv.json)**.
- The **[bible_index_niv](src/bible_search_engine/bible_index_niv)** directory is the stored Bible chapter index.
- The **[bible_queries_relevances](src/bible_search_engine/bible_queries_relevances)** directories contains the train queries relevance scores data **[train_queries_relevances.csv](src/bible_search_engine/bible_queries_relevances/train_queries_relevances.csv)** and the test queries relevance scores data **[test_queries_relevances.csv](src/bible_search_engine/bible_queries_relevances/test_queries_relevances.csv)**. The process to create them is in [`query_results.ipynb`](src/query_results.ipynb).
- The **[rankers](src/rankers)** directory contains the performances of the baselines and Bible search engine on the test data **[results.csv](src/rankers/results.csv)** and the bar plot of it **[results.jpg](src/rankers/results.jpg)**. The process to evaluate them is in [`evaluate_trained_ranker.ipynb`](src/evaluate_trained_ranker.ipynb).

## Release Notes
- Transitioned from King James Version to New International Version for Modern English
- Predicted Relevance from Annotated Queries Data for Completeness
- Incorporated Large Language Model for Chapter and Query Expansion
- Replaced Cross-Encoder with Colbert for Efficiency
- Tuned Hyperparameters for Optimal Ranking

## Planned Improvements
- Add Support for Different Bible Versions
- Use Online Learning

## References
Bible Study Tools. (2025). Retrieved from [https://www.biblestudytools.com/](https://www.biblestudytools.com)  

Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-strength Natural Language Processing in Python. doi:10.5281/zenodo.1212303

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., . . . Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In *Proceedings of the 31st International Conference on Neural Information Processing Systems* (pp. 3149–3157). Curran Associates Inc.  

Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics. Retrieved from [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)

Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., & Zaharia, M. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. Retrieved from [https://arxiv.org/abs/2112.01488](https://arxiv.org/abs/2112.01488)
