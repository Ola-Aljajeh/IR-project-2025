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

# Patch ONLY ir_datasets TSV reader to use utf-8
import ir_datasets.formats.tsv as tsv
import io

# Patch the part where text stream is decoded
original_iter = tsv.FileLineIter.__next__

def patched_next(self):
    if not hasattr(self, 'stream') or self.stream is None:
        raw = self.ctxt.enter_context(self.dlc.stream())
        self.stream = io.TextIOWrapper(raw, encoding='utf-8', errors='replace')
    return original_iter(self)

tsv.FileLineIter.__next__ = patched_next

import ir_datasets
import pandas as pd
from tqdm import tqdm
import os

# Load dataset
dataset = ir_datasets.load("antique")
csv_path = "antique_cleaned.csv"

# Check existing progress
processed_ids = set()
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path, usecols=['doc_id'])
        processed_ids = set(df['doc_id'].astype(str))
        print(f"🔁 Resuming from doc #{len(processed_ids)}")
    except Exception as e:
        print("⚠️ Could not load existing progress:", e)

# Create new file if needed
if not os.path.exists(csv_path):
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("doc_id,tokens,original_text\n")

# Preprocessing function
def preprocess(text):
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import string

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

# Main processing loop
docs = []
for doc in tqdm(dataset.docs_iter(), total=dataset.docs_count()):
    if doc.doc_id in processed_ids:
        continue

    try:
        text = doc.text or ""
        tokens = preprocess(text)

        docs.append({
            "doc_id": doc.doc_id,
            "tokens": tokens,
            "original_text": text.replace('\n', ' ').replace('\r', '')
        })

        if len(docs) >= 1000:
            pd.DataFrame(docs).to_csv(csv_path, mode='a', index=False, header=False, encoding='utf-8')
            docs = []

    except Exception as e:
        print(f"⚠️ Skipping doc {doc.doc_id}: {e}")
        continue

# Final batch
if docs:
    pd.DataFrame(docs).to_csv(csv_path, mode='a', index=False, header=False, encoding='utf-8')

print("✅ Done! Saved to antique_cleaned.csv")
