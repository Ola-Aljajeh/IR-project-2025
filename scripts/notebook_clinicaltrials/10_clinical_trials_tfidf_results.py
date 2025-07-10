import pandas as pd
import numpy as np
import joblib
import textwrap
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# === Fix for tokenizer ===
def whitespace_tokenizer(text):
    return text.split()

# === Load metadata ===
meta = pd.read_csv("clinical_trials_metadata.csv")
doc_ids = list(meta['doc_id'])
original_texts = list(meta['original_text'])

# === Load TF-IDF vectorizer and matrix ===
tfidf_vectorizer = joblib.load("clinical_trials_vectorizer.pkl")
tfidf_matrix = joblib.load("clinical_trials_tfidf_matrix.pkl")

# === Define your test queries ===
test_queries = [
    "What are the side effects of the vaccine?",
    "Who can participate in this study?",
    "How is the trial designed?",
    "Is the treatment safe?",
    "What are the eligibility criteria?"
]

# === Create TREC format output ===
output_path = "clinical_trials_tfidf_results.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for query_idx, query in enumerate(test_queries):
        query_vec = tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix)[0]
        top_indices = np.argsort(scores)[::-1][:100]  # top 100

        for rank, idx in enumerate(top_indices):
            f.write(f"{query_idx+1} Q0 {doc_ids[idx]} {rank+1} {scores[idx]:.4f} TFIDF\n")

print(f"✅ TF-IDF results saved to {output_path}")