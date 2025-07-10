import pandas as pd
import json
from collections import defaultdict

# Load preprocessed dataset
df = pd.read_csv("antique_cleaned.csv")
df['tokens'] = df['tokens'].apply(eval)

# Build inverted index
inverted_index = defaultdict(set)

for idx, row in df.iterrows():
    doc_id = row['doc_id']
    tokens = row['tokens']
    for token in set(tokens):  # Avoid duplicate tokens in same doc
        inverted_index[token].add(doc_id)

# Convert sets to sorted lists for JSON serialization
inverted_index = {term: sorted(list(doc_ids)) for term, doc_ids in inverted_index.items()}

# Save inverted index
with open("antique_inverted_index.json", "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, indent=2)

print("✅ Inverted index for ANTIQUE built and saved.")
