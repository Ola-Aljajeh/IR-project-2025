# ✅ build_bm25_clinical_trials.py
import pandas as pd
import joblib
from rank_bm25 import BM25Okapi

# Load cleaned clinical trials data
df = pd.read_csv("clinical_trials_cleaned.csv")

# Prepare corpus: list of token lists
corpus = df['tokens'].apply(eval).tolist()

# Build BM25 index
bm25 = BM25Okapi(corpus)

# Save index and metadata
joblib.dump(bm25, "clinical_trials_bm25_index.pkl")
joblib.dump(corpus, "clinical_trials_bm25_corpus.pkl")
df[['doc_id', 'original_text']].to_csv("clinical_trials_metadata.csv", index=False)

print("✅ BM25 index for Clinical Trials built and saved.")
