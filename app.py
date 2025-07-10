from flask import Flask, render_template, request, send_file, jsonify
import joblib
import pandas as pd
import sys
from io import BytesIO
from services.search_service import run_search
from services.rag_service import generate_rag_answer
from services.cluster_service import cluster_documents
from services.topic_detection_service import detect_topics
from evaluation.evaluate_cluster import evaluate_clustering_quality
from services.suggestion_service import get_suggestions
from utils.tokenizer import whitespace_tokenizer
from sentence_transformers import SentenceTransformer

app = Flask(__name__, template_folder='.')

# Register tokenizer for unpickling
sys.modules['__main__'].whitespace_tokenizer = whitespace_tokenizer

# Load models and data
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

clinical_meta = pd.read_csv("data/metadata/clinical_trials_metadata.csv")
clinical_vectorizer = joblib.load("models/tfidf/clinical_trials_vectorizer.pkl")
clinical_tfidf = joblib.load("models/tfidf/clinical_trials_tfidf_matrix.pkl")
clinical_bm25_corpus = joblib.load("models/bm25/clinical_trials_bm25_corpus.pkl")
clinical_bm25 = joblib.load("models/bm25/clinical_trials_bm25_model.pkl")
clinical_bert_embeddings = joblib.load("models/bert/clinical_trials_bert_embeddings.pkl")

antique_meta = pd.read_csv("data/metadata/antique_metadata.csv")
antique_vectorizer = joblib.load("models/tfidf/antique_vectorizer.pkl")
antique_tfidf = joblib.load("models/tfidf/antique_tfidf_matrix.pkl")
antique_bm25_corpus = joblib.load("models/bm25/antique_bm25_corpus.pkl")
antique_bm25 = joblib.load("models/bm25/antique_bm25_model.pkl")
antique_bert_embeddings = joblib.load("models/bert/antique_bert_embeddings.pkl")

# Load queries
clinical_queries_df = pd.read_csv("data/queries/clinical_trials_queries.txt", sep="\t", names=["query_id", "text"])
clinical_queries = clinical_queries_df["text"].tolist()
clinical_query_map = dict(zip(clinical_queries_df["text"], clinical_queries_df["query_id"].astype(str)))

antique_queries_df = pd.read_csv("data/queries/antique_queries.txt", sep="\t", names=["query_id", "text"])
antique_queries = antique_queries_df["text"].tolist()
antique_query_map = dict(zip(antique_queries_df["text"], antique_queries_df["query_id"].astype(str)))

# Search cache
search_cache = {
    "results": [],
    "query": "",
    "method": "",
    "dataset": "",
    "clustered_results": {},
    "k": 3,
    "eval_metrics": {},
    "cluster_labels": {},
    "view": "clustered"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results", methods=["POST"])
def results():
    query = request.form.get("query")
    method = request.form.get("method")
    dataset = request.form.get("dataset")
    k = int(request.form.get("k", 3))
    view = request.form.get("view", "clustered")

    print(f"🔍 [INFO] Method: {method}, Dataset: {dataset}, k={k}, view={view}")

    if method == "rag":
        results = generate_rag_answer(query, dataset)
        print(f"[DEBUG] RAG returned {len(results)} results")

        search_cache.update({
            "results": results,
            "query": query,
            "method": method,
            "dataset": dataset,
            "clustered_results": {},
            "k": k,
            "eval_metrics": {},
            "cluster_labels": {},
            "view": view
        })
        return render_template("results.html", query=query, dataset=dataset, method=method,
                               results=results, k=k, view=view, eval_metrics={}, clustered_results={},
                               cluster_labels={})

    # Set dataset-specific data
    if dataset == "clinical_trials":
        meta_df = clinical_meta
        tfidf_vectorizer = clinical_vectorizer
        tfidf_matrix = clinical_tfidf
        bm25_model = clinical_bm25
        bert_embeddings = clinical_bert_embeddings
        qrels_file = "data/qrels/clinical_trials_qrels.txt"
        query_map = clinical_query_map
    elif dataset == "antique":
        meta_df = antique_meta
        tfidf_vectorizer = antique_vectorizer
        tfidf_matrix = antique_tfidf
        bm25_model = antique_bm25
        bert_embeddings = antique_bert_embeddings
        qrels_file = "data/qrels/antique_qrels.txt"
        query_map = antique_query_map
    else:
        return "❌ Invalid dataset selected.", 400

    if method in {"tfidf", "bm25", "bert", "hybrid"}:
        results = run_search(query=query,
                             method=method,
                             dataset=dataset,
                             tfidf_vectorizer=tfidf_vectorizer,
                             tfidf_matrix=tfidf_matrix,
                             bm25_model=bm25_model,
                             bert_model=bert_model,
                             bert_embeddings=bert_embeddings,
                             tokenizer_fn=whitespace_tokenizer,
                             meta_df=meta_df)
        for result in results:
            try:
                result['score'] = float(result['score'])
            except (ValueError, TypeError):
                result['score'] = 0.0
    else:
        return "❌ Invalid retrieval method.", 400

    qrels = {}
    with open(qrels_file, "r", encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = int(rel)

    clustered = cluster_documents(results, k)
    cluster_labels = detect_topics(clustered)
    eval_metrics = evaluate_clustering_quality(results, clustered, qrels, query, query_map)

    search_cache.update({
        "results": results,
        "query": query,
        "method": method,
        "dataset": dataset,
        "clustered_results": clustered,
        "k": k,
        "eval_metrics": eval_metrics,
        "cluster_labels": cluster_labels,
        "view": view
    })

    return render_template("results.html",
                           query=query,
                           dataset=dataset,
                           method=method,
                           results=results,
                           clustered_results=clustered,
                           k=k,
                           view=view,
                           eval_metrics=eval_metrics,
                           cluster_labels=cluster_labels)

@app.route("/download")
def download():
    df = pd.DataFrame(search_cache["results"])
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="search_results.csv")

@app.route("/download_topics")
def download_topics():
    topics = search_cache.get("cluster_labels", {})
    topic_df = pd.DataFrame([
        {"Cluster ID": cid, "Topic Label": label, "Confidence": score}
        for cid, (label, score) in topics.items()
    ])
    output = BytesIO()
    topic_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="cluster_topics.csv")

@app.route("/suggest")
def suggest():
    partial = request.args.get("q", "").lower()
    dataset = request.args.get("dataset", "clinical_trials")
    matches = get_suggestions(partial, dataset, clinical_queries, antique_queries)
    return jsonify(matches)

if __name__ == "__main__":
    app.run(debug=True)
