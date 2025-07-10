# services/rag_service.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import joblib
import numpy as np

# Load FLAN-T5 once
rag_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
rag_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Load metadata
clinical_meta = pd.read_csv("C:/Users/Ola/Desktop/IR-project-2025/data/metadata/clinical_trials_metadata.csv")
antique_meta = pd.read_csv("C:/Users/Ola/Desktop/IR-project-2025/data/metadata/antique_metadata.csv")

# Whitespace tokenizer
def whitespace_tokenizer(text):
    return text.split()

def generate_rag_answer(query, dataset, top_k=3):
    """
    Retrieves top passages using BM25, then feeds them into a FLAN-T5 model for answer generation.
    """
    if dataset == "clinical_trials":
        meta = clinical_meta
        bm25_model = joblib.load("C:/Users/Ola/Desktop/IR-project-2025/models/bm25/clinical_trials_bm25_model.pkl")
    else:
        meta = antique_meta
        bm25_model = joblib.load("C:/Users/Ola/Desktop/IR-project-2025/models/bm25/antique_bm25_model.pkl")

    query_tokens = whitespace_tokenizer(query)
    bm25_scores = bm25_model.get_scores(query_tokens)
    top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    top_passages = meta.iloc[top_indices]["original_text"].tolist()

    # Combine context
    context = "\n".join(top_passages)
    prompt = f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Generate answer
    input_ids = rag_tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    output_ids = rag_model.generate(input_ids, max_new_tokens=100)
    answer = rag_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return [{"doc_id": "RAG-Answer", "score": "-", "text": answer}]
