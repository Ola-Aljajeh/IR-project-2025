import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load cleaned ANTIQUE data
df = pd.read_csv("antique_cleaned.csv")

# Use original text for embedding
texts = df['original_text'].tolist()

# Load BERT model (same as Clinical Trials for consistency)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode with batching
batch_size = 64
all_embeddings = []
for i in tqdm(range(0, len(texts), batch_size), desc="Embedding ANTIQUE"):
    batch = texts[i:i+batch_size]
    embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
    all_embeddings.extend(embeddings)

# Save embeddings and metadata
joblib.dump(all_embeddings, "antique_bert_embeddings.pkl")
df[['doc_id', 'original_text']].to_csv("antique_metadata.csv", index=False)

print("✅ BERT embeddings for ANTIQUE built and saved.")
