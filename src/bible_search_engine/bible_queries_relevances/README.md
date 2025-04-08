# Author
Ronen H  

# Time Frame
October 2023 to December 2023, April 2025 (predict relevance)

# Method
For each query:
1. Obtain top 50 chapters from ["msmarco-distilbert-dot-v5"](https://huggingface.co/sentence-transformers/msmarco-distilbert-dot-v5) bi-encoder.
2. Relevance label each chapter on a scale of 1 (least) to 5 (most).
3. Add chapters to provide more examples if necessary.
4. Fine-Tune ["allenai/longformer-base-4096"](https://huggingface.co/allenai/longformer-base-4096) to predict relevance for rest of chapters.

# Fine-Tuned Relevance Classifier
The fine-tuned relevance classifier can be seen on Hugging Face at [https://huggingface.co/ronenh24/longformer-base-4096-bible](https://huggingface.co/ronenh24/longformer-base-4096-bible).
