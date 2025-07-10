import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

# Fix: define the tokenizer used during vectorizer creation
def whitespace_tokenizer(text):
    return text.split()

# Step 1: Load saved TF-IDF components
vectorizer = joblib.load("antique_vectorizer.pkl")
tfidf_matrix = joblib.load("antique_tfidf_matrix.pkl")
metadata = pd.read_csv("antique_metadata.csv")

# Step 2: Define test queries
queries = [
    "How do I reset my Gmail password?",
    "What is the average salary for a registered nurse?",
    "Can I take ibuprofen and paracetamol together?",
    "What are the side effects of quitting smoking?",
    "How to apply for a passport online?"
]

# Step 3: Search function
def search(query, top_k=5):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    results = metadata.iloc[top_indices].copy()
    results['score'] = scores[top_indices]
    return results

# Step 4: Run and print results
for q in queries:
    print("="*30, "\n🔍 Query:", q)
    results = search(q)
    for i, row in results.iterrows():
        print(f"\n🔹 Doc ID: {row['doc_id']}  (Score: {row['score']:.4f})\n{textwrap.fill(row['original_text'], 100)}")
