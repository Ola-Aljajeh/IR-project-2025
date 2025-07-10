# build_bert_embeddings_clinical.py

import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

# Load preprocessed data
df = pd.read_csv("clinical_trials_cleaned.csv")
df['joined'] = df['tokens'].apply(eval).apply(lambda tokens: ' '.join(tokens))

# Load a compact, efficient BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")  # ~384D embeddings

# Encode documents in batches
batch_size = 256
all_embeddings = []
for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df['joined'].iloc[i:i+batch_size].tolist()
    embeddings = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
    all_embeddings.append(embeddings)

# Stack all batches into one array
all_embeddings = np.vstack(all_embeddings)

# Save embeddings and doc IDs
joblib.dump(all_embeddings, "clinical_trials_bert_embeddings.pkl")
df[['doc_id', 'original_text']].to_csv("clinical_trials_metadata.csv", index=False)

print("✅ BERT embeddings for Clinical Trials built and saved.")
