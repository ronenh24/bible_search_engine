# Author: Ronen Huang

import pandas as pd
import os


def get_train_test_queries(query_dir: str="", num_queries: int=20, use_pred: bool=False) -> None:
    """
    query_dir: Where to store train and test query relevance data. Default to current directory.
    num_queries: Number of queries to split. Defaults to 20.
    use_pred: Whether to use predicted relevance label or not. Defaults to not.

    Individual query data have names "query#.csv" (or "query#_pred.csv" if use predicted) from 1 to number of queries.
    """
    suffix = ".csv"
    if use_pred:
        suffix = "_pred" + suffix

    train_df = pd.DataFrame()
    train_end = num_queries // 2
    for i in range(1, train_end + 1):
        query_path = query_dir + "/query" + str(i) + suffix
        if not os.path.isfile(query_path):
            raise Exception("Query data " + query_path + " does not exist.")
        train_df = pd.concat([train_df, pd.read_csv(query_path)], ignore_index=True)

    test_df = pd.DataFrame()
    test_start = train_end + 1
    for i in range(test_start, num_queries + 1):
        query_path = query_dir + "/query" + str(i) + suffix
        if not os.path.isfile(query_path):
            raise Exception("Query data " + query_path + " does not exist.")
        test_df = pd.concat([test_df, pd.read_csv(query_path)], ignore_index=True)

    train_df.to_csv(query_dir + "/train_queries_relevances.csv", index=None)
    test_df.to_csv(query_dir + "/test_queries_relevances.csv", index=None)
