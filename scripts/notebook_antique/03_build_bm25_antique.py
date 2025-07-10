import pandas as pd
from rank_bm25 import BM25Okapi
import joblib
import ast

# Load cleaned ANTIQUE dataset
df = pd.read_csv("antique_cleaned.csv")

# Convert 'tokens' column from string to list
df['tokens'] = df['tokens'].apply(ast.literal_eval)

# Build BM25
tokenized_corpus = df['tokens'].tolist()
bm25 = BM25Okapi(tokenized_corpus)

# Save the model and supporting data
joblib.dump(tokenized_corpus, "antique_bm25_corpus.pkl")
joblib.dump(bm25, "antique_bm25_model.pkl")
df[['doc_id', 'original_text']].to_csv("antique_metadata.csv", index=False)

print("✅ BM25 index for ANTIQUE built and saved.")
