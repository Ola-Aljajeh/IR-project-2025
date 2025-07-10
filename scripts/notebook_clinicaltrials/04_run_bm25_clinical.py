import pandas as pd
import joblib
from rank_bm25 import BM25Okapi
import textwrap

# Load data
with open("clinical_trials_bm25_corpus.pkl", "rb") as f:
    corpus = joblib.load(f)
with open("clinical_trials_bm25_model.pkl", "rb") as f:
    bm25 = joblib.load(f)
df_meta = pd.read_csv("clinical_trials_metadata.csv")

# Helper to show results
def run_query(query, top_k=5):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for i in top_idx:
        doc_id = df_meta.iloc[i]['doc_id']
        text = df_meta.iloc[i]['original_text']
        results.append((doc_id, scores[i], text))
    return results

# Define test queries
queries = {
    "TREATMENT SAFETY": [
        "What are the side effects of the vaccine?",
        "Is the treatment well tolerated by patients?",
        "Any serious adverse events reported during the trial?"
    ],
    "PATIENT ELIGIBILITY": [
        "Who can participate in the trial?",
        "Eligibility criteria for patients with lung cancer",
        "Am I eligible for this study?"
    ],
    "TRIAL DESIGN": [
        "How is the study structured?",
        "What is the primary endpoint of the trial?",
        "How long does the clinical trial last?"
    ]
}

# Run and print results
for section, qs in queries.items():
    print("="*30)
    print(f"🧪 {section}")
    print("="*30)
    for query in qs:
        print(f"\n🔍 Query: {query}\n")
        results = run_query(query)
        for doc_id, score, text in results:
            print(f"🔹 Doc ID: {doc_id}  (Score: {score:.4f})")
            print(textwrap.fill(text, width=100))
            print()
