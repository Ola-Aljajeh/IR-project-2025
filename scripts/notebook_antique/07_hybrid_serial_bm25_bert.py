# build_hybrid_antique.py
import pandas as pd
import joblib
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load cleaned ANTIQUE data
df = pd.read_csv("antique_cleaned.csv")

# Tokenize for BM25
tokenized_corpus = [eval(doc) for doc in df['tokens']]
bm25 = BM25Okapi(tokenized_corpus)

# Save BM25
joblib.dump(bm25, "antique_bm25_model.pkl")
joblib.dump(tokenized_corpus, "antique_bm25_corpus.pkl")

# Load BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert token lists back to text
texts = [' '.join(tokens) for tokens in tokenized_corpus]

# Compute BERT embeddings
bert_embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
joblib.dump(bert_embeddings, "antique_bert_embeddings.pkl")

# Save metadata
df[['doc_id', 'original_text']].to_csv("antique_metadata.csv", index=False)

print("✅ Hybrid BM25 + BERT model for ANTIQUE dataset built and saved.")

import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import textwrap

# Load preprocessed tokens, metadata, and hybrid components
with open("antique_bm25_corpus.pkl", "rb") as f:
    corpus = joblib.load(f)

with open("antique_bm25_model.pkl", "rb") as f:
    bm25 = joblib.load(f)

doc_embeddings = joblib.load("antique_bert_embeddings.pkl")
df_meta = pd.read_csv("antique_metadata.csv")

# Map from doc_id to index (for embedding lookup)
doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(df_meta['doc_id'])}

# Load the same BERT model used for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define your test queries
queries = [
    "How do I reset my Gmail password?",
    "What is the average salary for a registered nurse?",
    "Can I take ibuprofen and paracetamol together?",
    "What are the side effects of quitting smoking?",
    "How to apply for a passport online?"
]

def hybrid_search(query, top_k=5, bm25_k=20, alpha=0.6):
    # Step 1: BM25 search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_k]

    # Step 2: BERT query embedding
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Step 3: BERT similarity with top BM25 docs
    top_doc_embeddings = np.array([doc_embeddings[i] for i in top_bm25_indices])
    bert_scores = util.cos_sim(query_embedding, top_doc_embeddings)[0].cpu().numpy()

    # Combine scores
    hybrid_scores = alpha * bm25_scores[top_bm25_indices] + (1 - alpha) * bert_scores

    # Rank final top_k
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    results = []
    for rank in top_indices:
        doc_idx = top_bm25_indices[rank]
        results.append({
            "doc_id": df_meta.iloc[doc_idx]['doc_id'],
            "score": hybrid_scores[rank],
            "text": df_meta.iloc[doc_idx]['original_text']
        })
    return results

# Run and print results
for query in queries:
    print("="*30)
    print(f"🔍 Query: {query}\n")
    results = hybrid_search(query)
    for res in results:
        print(f"🔹 Doc ID: {res['doc_id']}  (Score: {res['score']:.4f})")
        print(textwrap.fill(res['text'], 100))
        print()
print("✅ All hybrid queries on ANTIQUE processed.")
