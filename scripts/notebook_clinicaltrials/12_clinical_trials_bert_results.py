import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, util

# === Load document metadata ===
df_meta = pd.read_csv("clinical_trials_metadata.csv")
doc_ids = list(df_meta['doc_id'])
original_texts = list(df_meta['original_text'])

# === Load BERT model and document embeddings ===
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = joblib.load("clinical_trials_bert_embeddings.pkl")  # or .npy if that's what you saved

# === Define test queries ===
test_queries = [
    "What are the side effects of the vaccine?",
    "Who can participate in this study?",
    "How is the trial designed?",
    "Is the treatment safe?",
    "What are the eligibility criteria?"
]

# === Compute and save results in TREC format ===
output_path = "clinical_trials_bert_results.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for query_idx, query in enumerate(test_queries):
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:100]  # top 100

        for rank, idx in enumerate(top_indices):
            f.write(f"{query_idx+1} Q0 {doc_ids[idx]} {rank+1} {scores[idx]:.4f} BERT\n")

print(f"✅ BERT results saved to {output_path}")