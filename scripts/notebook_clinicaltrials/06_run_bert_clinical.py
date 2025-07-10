import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

# Load precomputed embeddings and metadata
doc_embeddings = joblib.load("clinical_trials_bert_embeddings.pkl")
df_meta = pd.read_csv("clinical_trials_metadata.csv")

# Load the same BERT model used during building
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example queries
queries = [
    "What are the side effects of the vaccine?",
    "Who can participate in the trial?",
    "What is the design of the clinical trial?"
]

# Number of results to retrieve per query
TOP_K = 5

for query in queries:
    print("=" * 60)
    print(f"🔍 Query: {query}\n")

    # Encode query
    query_embedding = model.encode([query])

    # Compute cosine similarities
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Get top-k indices
    top_k_idx = np.argsort(scores)[::-1][:TOP_K]

    for idx in top_k_idx:
        doc_id = df_meta.iloc[idx]['doc_id']
        text = df_meta.iloc[idx]['original_text']
        score = scores[idx]
        print(f"🔹 Doc ID: {doc_id}  (Score: {score:.4f})\n{textwrap.fill(text, width=100)}\n")
