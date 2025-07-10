import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

import ir_datasets
import pandas as pd
from tqdm import tqdm
import os

# Load ClinicalTrials dataset
dataset = ir_datasets.load("clinicaltrials/2017")
csv_path = "clinical_trials_cleaned.csv"

# Create the CSV file with headers if it doesn't exist
if not os.path.exists(csv_path):
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("doc_id,tokens,original_text\n")

# Load already processed doc_ids
processed_ids = set()
if os.path.exists(csv_path):
    try:
        df_existing = pd.read_csv(csv_path, usecols=['doc_id'])
        processed_ids = set(df_existing['doc_id'].astype(str))
        print(f"🔁 Resuming from doc #{len(processed_ids)}")
    except Exception as e:
        print("⚠️ Couldn't read existing file correctly. Starting from scratch.")

# Process and save in chunks
docs = []
for doc in tqdm(dataset.docs_iter(), total=dataset.docs_count()):
    if doc.doc_id in processed_ids:
        continue

    # Combine fields (title + summary + detailed_description + eligibility)
    text = f"{doc.title or ''} {doc.summary or ''} {doc.detailed_description or ''} {doc.eligibility or ''}"

    tokens = preprocess(text)

    docs.append({
        "doc_id": doc.doc_id,
        "tokens": tokens,
        "original_text": text.replace('\n', ' ').replace('\r', '')
    })

    # Save every 1000 documents
    if len(docs) >= 1000:
        pd.DataFrame(docs).to_csv(csv_path, mode='a', index=False, header=False)
        docs = []

# Save the final batch
if docs:
    pd.DataFrame(docs).to_csv(csv_path, mode='a', index=False, header=False)

print("✅ All done! Saved to clinical_trials_cleaned.csv")
