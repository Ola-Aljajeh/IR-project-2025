import pandas as pd
import joblib
from rank_bm25 import BM25Okapi

# Load data
df_meta = pd.read_csv("clinical_trials_metadata.csv")
queries = pd.read_csv("clinical_trials_queries.txt", sep="\t", names=["query_id", "text"])

# Load BM25 model and corpus
with open("clinical_trials_bm25_corpus.pkl", "rb") as f:
    corpus = joblib.load(f)
with open("clinical_trials_bm25_model.pkl", "rb") as f:
    bm25 = joblib.load(f)

doc_ids = df_meta['doc_id'].tolist()

with open("clinical_trials_bm25_results.txt", "w", encoding="utf-8") as f:
    for _, row in queries.iterrows():
        query_id, query_text = row['query_id'], row['text']
        tokenized_query = query_text.lower().split()
        scores = bm25.get_scores(tokenized_query)
        ranked_indices = scores.argsort()[::-1]

        for rank, idx in enumerate(ranked_indices[:1000]):
            f.write(f"{query_id} Q0 {doc_ids[idx]} {rank+1} {scores[idx]:.4f} BM25\n")

print("✅ BM25 results saved to clinical_trials_bm25_results.txt")