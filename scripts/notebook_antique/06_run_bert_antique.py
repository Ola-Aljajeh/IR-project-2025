import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

# Load precomputed BERT document embeddings and metadata
doc_embeddings = joblib.load("antique_bert_embeddings.pkl")
df_meta = pd.read_csv("antique_metadata.csv")

# Load the same BERT model used during indexing
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example test queries
queries = [
    "How do I reset my Gmail password?",
    "What is the average salary for a registered nurse?",
    "Can I take ibuprofen and paracetamol together?",
    "What are the side effects of quitting smoking?",
    "How to apply for a passport online?"
]

# Number of results to retrieve per query
TOP_K = 5

# Run queries
for query in queries:
    print("=" * 60)
    print(f"🔍 Query: {query}\n")

    # Encode the query
    query_embedding = model.encode([query])

    # Compute cosine similarity with all documents
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Get top-k documents
    top_k_idx = np.argsort(scores)[::-1][:TOP_K]

    for idx in top_k_idx:
        doc_id = df_meta.iloc[idx]['doc_id']
        text = df_meta.iloc[idx]['original_text']
        score = scores[idx]
        print(f"🔹 Doc ID: {doc_id}  (Score: {score:.4f})\n{textwrap.fill(text, width=100)}\n")
