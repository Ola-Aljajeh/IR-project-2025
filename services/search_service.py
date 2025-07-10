# services/search_service.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util

def run_search(
    query,
    method,
    dataset,
    tfidf_vectorizer,
    tfidf_matrix,
    bm25_model,
    bert_model,
    bert_embeddings,
    tokenizer_fn,
    meta_df
):
    """
    Run one of the four retrieval methods (TF-IDF, BM25, BERT, Hybrid)
    and return top 10 results formatted.
    """
    query_tokens = tokenizer_fn(query)

    if method == "tfidf":
        query_vec = tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    elif method == "bm25":
        scores = bm25_model.get_scores(query_tokens)

    elif method == "bert":
        query_embedding = bert_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, bert_embeddings)[0].cpu().numpy()

    elif method == "hybrid":
        # Combine BM25 and BERT
        bm25_scores = bm25_model.get_scores(query_tokens)
        query_embedding = bert_model.encode(query, convert_to_tensor=True)
        bert_scores = util.cos_sim(query_embedding, bert_embeddings)[0].cpu().numpy()
        scores = 0.5 * np.array(bm25_scores) + 0.5 * np.array(bert_scores)

    else:
        raise ValueError("Unsupported retrieval method")

    # Sort and get top results
    top_indices = np.argsort(scores)[::-1][:10]
    top_docs = meta_df.iloc[top_indices]
    top_scores = scores[top_indices]

    results = [
        {
            "doc_id": doc_id,
            "score": f"{score:.4f}",
            "text": text[:500] + "..."
        }
        for doc_id, score, text in zip(top_docs["doc_id"], top_scores, top_docs["original_text"])
    ]

    return results
