import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

# === Load document metadata and tokens ===
df = pd.read_csv("clinical_trials_cleaned.csv")
df['tokens'] = df['tokens'].apply(eval)
df_meta = pd.read_csv("clinical_trials_metadata.csv")
doc_ids = list(df_meta['doc_id'])

# === BM25 setup ===
tokenized_corpus = list(df['tokens'])
bm25 = BM25Okapi(tokenized_corpus)

# === BERT setup ===
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = joblib.load("clinical_trials_bert_embeddings.pkl")

# === Define test queries ===
test_queries = [
    "What are the side effects of the vaccine?",
    "Who can participate in this study?",
    "How is the trial designed?",
    "Is the treatment safe?",
    "What are the eligibility criteria?"
]

# === Hybrid parameters ===
alpha = 0.5  # weight between BM25 and BERT scores

# === Save results in TREC format ===
output_path = "clinical_trials_hybrid_results.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for query_idx, query in enumerate(test_queries):
        query_embedding = model.encode(query, convert_to_tensor=True)
        bert_scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()

        bm25_scores = bm25.get_scores(query.split())
        bm25_scores = np.array(bm25_scores)

        # Normalize both
        norm_bm25 = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        norm_bert = (bert_scores - bert_scores.min()) / (bert_scores.max() - bert_scores.min() + 1e-8)

        hybrid_scores = alpha * norm_bm25 + (1 - alpha) * norm_bert
        top_indices = np.argsort(hybrid_scores)[::-1][:100]

        for rank, idx in enumerate(top_indices):
            f.write(f"{query_idx+1} Q0 {doc_ids[idx]} {rank+1} {hybrid_scores[idx]:.4f} HYBRID\n")

print(f"✅ Hybrid results saved to {output_path}")