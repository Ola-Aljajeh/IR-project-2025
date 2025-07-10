import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# 🔧 Define custom tokenizer used during TF-IDF indexing
def whitespace_tokenizer(text):
    return text.split()

# 📂 Load metadata and TF-IDF components
df_meta = pd.read_csv("clinical_trials_metadata.csv")
tfidf_vectorizer = joblib.load("clinical_trials_vectorizer.pkl")
tfidf_matrix = joblib.load("clinical_trials_tfidf_matrix.pkl")

# 🔍 Load official queries
queries = pd.read_csv("clinical_trials_queries.txt", sep='\t', names=["query_id", "text"])

# 💾 Write ranked results to file
with open("clinical_trials_tfidf_results.txt", "w", encoding="utf-8") as f:
    for _, row in queries.iterrows():
        query_id, text = row["query_id"], row["text"]
        query_vec = tfidf_vectorizer.transform([text])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:1000]
        for rank, idx in enumerate(top_indices, start=1):
            doc_id = df_meta.iloc[idx]["doc_id"]
            f.write(f"{query_id} Q0 {doc_id} {rank} {scores[idx]:.4f} TFIDF\n")

print("✅ TF-IDF results saved to clinical_trials_tfidf_results.txt")