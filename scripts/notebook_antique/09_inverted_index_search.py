import json
import pandas as pd
from collections import Counter
import textwrap

# Load inverted index and metadata
with open("antique_inverted_index.json", "r", encoding="utf-8") as f:
    inverted_index = json.load(f)

df_meta = pd.read_csv("antique_cleaned.csv")
df_meta['original_text'] = df_meta['original_text'].astype(str)
doc_map = dict(zip(df_meta['doc_id'], df_meta['original_text']))

def search(query, top_k=5):
    terms = query.lower().split()
    doc_scores = Counter()

    for term in terms:
        doc_ids = inverted_index.get(term, [])
        for doc_id in doc_ids:
            doc_scores[doc_id] += 1

    top_docs = doc_scores.most_common(top_k)
    return [(doc_id, doc_map.get(doc_id, ""), score) for doc_id, score in top_docs]

# Example queries
queries = [
    "gmail password reset",
    "side effects of ibuprofen",
    "apply for passport"
]

for query in queries:
    print("="*80)
    print(f"🔍 Query: {query}")
    results = search(query)
    for doc_id, text, score in results:
        print(f"\n🔹 Doc ID: {doc_id}  (Score: {score})")
        print(textwrap.fill(text, 100))
