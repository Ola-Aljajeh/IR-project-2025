import pandas as pd
import numpy as np
import joblib
from rank_bm25 import BM25Okapi

# === Load metadata ===
df = pd.read_csv("clinical_trials_metadata.csv")
doc_ids = list(df['doc_id'])
original_texts = list(df['original_text'])

# === Load preprocessed corpus ===
with open("clinical_trials_bm25_corpus.pkl", "rb") as f:
    corpus = joblib.load(f)

# === Load BM25 model ===
with open("clinical_trials_bm25_model.pkl", "rb") as f:
    bm25 = joblib.load(f)

# === Define test queries ===
test_queries = [
    "What are the side effects of the vaccine?",
    "Who can participate in this study?",
    "How is the trial designed?",
    "Is the treatment safe?",
    "What are the eligibility criteria?"
]

# === Create TREC format output ===
output_path = "clinical_trials_bm25_results.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for query_idx, query in enumerate(test_queries):
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:100]  # top 100

        for rank, idx in enumerate(top_indices):
            f.write(f"{query_idx+1} Q0 {doc_ids[idx]} {rank+1} {scores[idx]:.4f} BM25\n")

print(f"✅ BM25 results saved to {output_path}")