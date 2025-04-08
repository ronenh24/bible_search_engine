# Author: Ronen Huang

from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding,
                          EvalPrediction, TrainingArguments, Trainer)
from datasets import Dataset
import pandas as pd
import orjson
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import torch
from torchvision.ops import sigmoid_focal_loss
import os


class PredictRelevance:
    """
    Predicts rest of relevance using annotated queries.
    """
    def __init__(self, testament_paths: list[str], model_name: str = "allenai/longformer-base-4096",
                 max_length: int = 4096) -> None:
        """
        testament_paths: Paths to old testament chapters and new testament chapters data.
        model_name: Model to use for sequence classification.
        max_length: Maximum number of tokens to consider.
        """
        self.testament_paths = testament_paths
        self.model_name = model_name
        self.output_dir = self.model_name.split("/")[-1].replace("-bible", "") + "-bible"
        self.max_length = max_length
        self.chapter_text_df = self.get_chapter_text_df(testament_paths)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5,
                                                                        problem_type="single_label_classification")

    def get_chapter_text_df(self) -> pd.DataFrame:
        """
        Get the text of each chapter.
        """
        chapter_list = []
        chapterid_list = []
        chaptertext_list = []
        for testament_path in self.testament_paths:
            with open(testament_path, 'rb') as testament_file:
                for chapter_line in testament_file:
                    chapter = orjson.loads(chapter_line)
                    i = 1
                    chapter_list.append(chapter["chapter"])
                    chapterid_list.append(chapter["chapterid"])
                    chaptertext_list.append(" ".join(chapter["verses"].values()))
        return pd.DataFrame({"chapter": chapter_list, "chapterid": chapterid_list,
                             "chaptertext": chaptertext_list})

    def get_query_df(self, query_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        query_path: Path to annotated query relevances.

        Get train data of annotated data and test data to annotate.
        """
        query_df = pd.read_csv(query_path)
        query_df = query_df.merge(self.chapter_text_df, "outer")
        query_df = query_df.rename(columns={"relevance": "label"})
        query_df["label"] = query_df["label"] - 1

        train_query_df = query_df[query_df["label"].notna()]
        train_query_df["label"] = train_query_df["label"].astype(int)

        test_query_df = query_df[query_df["label"].isna()]
        test_query_df = test_query_df.drop(columns="label")
        test_query_df["query"] = train_query_df["query"].unique()[0]

        return train_query_df, test_query_df

    def tokenize(self, batch: Dataset) -> BatchEncoding:
        return self.tokenizer(batch["chaptertext"], padding="longest", truncation=True,
                              max_length=self.max_length, return_tensors="pt")

    def compute_loss_func(self, outputs: BatchEncoding, labels: torch.Tensor,
                          num_items_in_batch: int = None) -> torch.Tensor:
        logits = outputs.get('logits')
        labels = torch.nn.functional.one_hot(labels, 5).float()
        return sigmoid_focal_loss(logits, labels, reduction="mean").to("cuda")

    def compute_metrics(self, eval_pred: EvalPrediction) -> dict[str, float]:
        logits = eval_pred.predictions
        actual = label_binarize(eval_pred.label_ids, classes=[0, 1, 2, 3, 4])

        macro_ap = average_precision_score(actual, logits, average="macro")
        weighted_ap = average_precision_score(actual, logits, average="weighted")

        return {"macro_average_precision": macro_ap, "weighted_average_precision": weighted_ap}

    def train(self, query_path: str, learning_rate: float=3e-5) -> None:
        """
        query_path: Path to annotated query relevances.

        Predicts rest of relevance based on annotated query data.
        """
        if not os.path.isfile(query_path):
            raise Exception("Query data " + query_path + " does not exist.")

        train_query_df, test_query_df = self.get_query_df(query_path)

        warmup_steps = 4 * (train_query_df.shape[0] // 8 + int(train_query_df.shape[0] % 8 != 0))

        train_query_dataset = Dataset.from_pandas(train_query_df).map(self.tokenize, batched=True)
        test_query_dataset = Dataset.from_pandas(test_query_df).map(self.tokenize, batched=True)

        training_args = TrainingArguments(output_dir=self.output_dir, overwrite_output_dir=True,
                                          learning_rate=learning_rate, num_train_epochs=12, do_eval=True,
                                          logging_strategy="epoch", eval_strategy="epoch", save_strategy="epoch",
                                          save_total_limit=1, load_best_model_at_end=True,
                                          metric_for_best_model="eval_macro_average_precision",
                                          greater_is_better=True, lr_scheduler_type="cosine_with_restarts",
                                          lr_scheduler_kwargs={"num_cycles": 2}, warmup_steps=warmup_steps,
                                          report_to="none", push_to_hub=True, auto_find_batch_size=True)
        trainer = Trainer(model=self.model, args=training_args, train_dataset=train_query_dataset,
                          eval_dataset=train_query_dataset, processing_class=self.tokenizer,
                          compute_loss_func=self.compute_loss_func, compute_metrics=self.compute_metrics)
        trainer.train()
        trainer.push_to_hub("End of training for query " + train_query_df["query"].unique()[0])

        test_query_df["label"] = np.argmax(trainer.predict(test_query_dataset).predictions, 1) + 1
        test_query_df = test_query_df.rename(columns={"label": "relevance"})
        test_query_df = test_query_df.drop(columns="chaptertext")
        train_query_df["label"] = train_query_df["label"] + 1
        train_query_df = pd.read_csv(query_path)
        pd.concat([train_query_df, test_query_df]).to_csv(query_path.split("/")[-1].replace(".csv", "_pred.csv"),
                                                          index=None)

        self.__init__(self.testament_paths, self.output_dir, self.max_length)
