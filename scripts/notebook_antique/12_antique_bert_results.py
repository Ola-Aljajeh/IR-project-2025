import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df_meta = pd.read_csv("clinical_trials_metadata.csv")
queries = pd.read_csv("clinical_trials_queries.txt", sep="\t", names=["query_id", "text"])
doc_embeddings = joblib.load("clinical_trials_bert_embeddings.pkl")

# Load BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

with open("clinical_trials_bert_results.txt", "w", encoding="utf-8") as f:
    for _, row in queries.iterrows():
        query_id, query_text = row['query_id'], row['text']
        query_embedding = model.encode([query_text])
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        ranked_indices = np.argsort(similarities)[::-1]

        for rank, idx in enumerate(ranked_indices[:1000]):
            f.write(f"{query_id} Q0 {df_meta['doc_id'].iloc[idx]} {rank+1} {similarities[idx]:.4f} BERT\n")

print("✅ BERT results saved to clinical_trials_bert_results.txt")