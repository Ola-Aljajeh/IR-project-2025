import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import minmax_scale

# Load data
df_meta = pd.read_csv("clinical_trials_metadata.csv")
queries = pd.read_csv("clinical_trials_queries.txt", sep="\t", names=["query_id", "text"])

# Load BM25
with open("clinical_trials_bm25_corpus.pkl", "rb") as f:
    corpus = joblib.load(f)
with open("clinical_trials_bm25_model.pkl", "rb") as f:
    bm25 = joblib.load(f)

# Load BERT
doc_embeddings = joblib.load("clinical_trials_bert_embeddings.pkl")
model = SentenceTransformer('all-MiniLM-L6-v2')

doc_ids = df_meta['doc_id'].tolist()

with open("clinical_trials_hybrid_results.txt", "w", encoding="utf-8") as f:
    for _, row in queries.iterrows():
        query_id, query_text = row['query_id'], row['text']
        tokenized_query = query_text.lower().split()
        query_embedding = model.encode([query_text])

        bm25_scores = bm25.get_scores(tokenized_query)
        cosine_scores = np.dot(doc_embeddings, query_embedding[0])

        # Normalize scores
        bm25_norm = minmax_scale(bm25_scores)
        cosine_norm = minmax_scale(cosine_scores)

        hybrid_scores = 0.5 * bm25_norm + 0.5 * cosine_norm
        ranked_indices = np.argsort(hybrid_scores)[::-1]

        for rank, idx in enumerate(ranked_indices[:1000]):
            f.write(f"{query_id} Q0 {doc_ids[idx]} {rank+1} {hybrid_scores[idx]:.4f} Hybrid\n")

print("✅ Hybrid results saved to clinical_trials_hybrid_results.txt")