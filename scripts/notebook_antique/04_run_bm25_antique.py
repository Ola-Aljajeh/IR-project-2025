import pandas as pd
import joblib
from rank_bm25 import BM25Okapi
import numpy as np
import textwrap

# Load BM25 index components
with open("antique_bm25_corpus.pkl", "rb") as f:
    tokenized_corpus = joblib.load(f)

with open("antique_bm25_model.pkl", "rb") as f:
    bm25 = joblib.load(f)

df_meta = pd.read_csv("antique_metadata.csv")

# Define queries to test
queries = [
    "How do I reset my Gmail password?",
    "What is the average salary for a registered nurse?",
    "Can I take ibuprofen and paracetamol together?",
    "What are the side effects of quitting smoking?",
    "How to apply for a passport online?"
]

# Tokenizer
def tokenize(text):
    return text.lower().split()

# Run queries
for q in queries:
    print("="*30)
    print(f"🔍 Query: {q}\n")
    tokenized_query = tokenize(q)
    scores = bm25.get_scores(tokenized_query)

    top_n = np.argsort(scores)[::-1][:5]

    for idx in top_n:
        doc_id = df_meta.iloc[idx]['doc_id']
        text = df_meta.iloc[idx]['original_text']
        score = scores[idx]
        print(f"🔹 Doc ID: {doc_id}  (Score: {score:.4f})")
        print(textwrap.fill(text, width=100))
        print()

print("✅ All BM25 queries processed.")
