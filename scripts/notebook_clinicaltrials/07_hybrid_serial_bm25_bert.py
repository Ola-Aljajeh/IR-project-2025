import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import textwrap

# Load preprocessed corpus (tokenized)
with open("clinical_trials_bm25_corpus.pkl", "rb") as f:
    corpus = joblib.load(f)

# Load BM25 model
with open("clinical_trials_bm25_model.pkl", "rb") as f:
    bm25 = joblib.load(f)

# Load original texts
df_meta = pd.read_csv("clinical_trials_metadata.csv")
doc_id_to_text = dict(zip(df_meta['doc_id'], df_meta['original_text']))

# Load BERT model and document embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = joblib.load("clinical_trials_bert_embeddings.pkl")

# Map from doc_id to index (to locate BERT embeddings)
doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(df_meta['doc_id'])}

# Example queries
queries = [
    "What are the side effects of the vaccine?",
    "Who can participate in the trial?",
    "What is the trial design for this cancer study?",
]

# Parameters
bm25_top_k = 100   # How many BM25 docs to consider
final_top_k = 5    # Final results to show after BERT reranking

for query in queries:
    print("=" * 40)
    print(f"🔍 Query: {query}\n")

    # Step 1: BM25 retrieval
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[::-1][:bm25_top_k]
    top_doc_ids = [df_meta.iloc[idx]['doc_id'] for idx in top_indices]

    # Step 2: BERT rerank on top BM25 docs
    top_texts = [doc_id_to_text[doc_id] for doc_id in top_doc_ids]
    top_indices_for_bert = [doc_id_to_index[doc_id] for doc_id in top_doc_ids]
    candidate_embeddings = doc_embeddings[top_indices_for_bert]

    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, candidate_embeddings)[0].cpu().numpy()

    reranked = sorted(zip(top_doc_ids, cos_scores), key=lambda x: x[1], reverse=True)[:final_top_k]

    for doc_id, score in reranked:
        wrapped_text = textwrap.fill(doc_id_to_text[doc_id], 100)
        print(f"🔹 Doc ID: {doc_id}  (Score: {score:.4f})\n{wrapped_text}\n")

print("✅ Hybrid BM25 + BERT reranking done.")