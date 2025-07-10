import pandas as pd
import joblib
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define the tokenizer used when creating the vectorizer
def whitespace_tokenizer(text):
    return text.split()

# Load vectorizer and matrix
vectorizer = joblib.load("clinical_trials_vectorizer.pkl")
tfidf_matrix = joblib.load("clinical_trials_tfidf_matrix.pkl")
metadata = pd.read_csv("clinical_trials_metadata.csv")

# Define search function
def search(query, top_k=3):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    results = metadata.iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    return results

# Define query categories
queries = {
    "Treatment Safety": [
        "What are the side effects of the treatment?",
        "Is the treatment well tolerated by patients?",
        "Any serious adverse events reported during the trial?"
    ],
    "Patient Eligibility": [
        "Who can participate in the trial?",
        "Eligibility criteria for patients with lung cancer",
        "Can children join the study?"
    ],
    "Trial Design": [
        "What is the study design?",
        "How is the control group treated?",
        "What are the primary endpoints of the trial?"
    ]
}

# Run queries
for category, qs in queries.items():
    print(f"\n{'='*25} {category.upper()} {'='*25}\n")
    for q in qs:
        print(f"🔍 Query: {q}\n")
        results = search(q, top_k=3)
        for _, row in results.iterrows():
            print(f"🔹 Doc ID: {row['doc_id']}  (Score: {row['score']:.4f})\n{textwrap.fill(row['original_text'], 100)}\n")

print("✅ All queries processed.")
