import pytrec_eval
import pandas as pd

# === Load QRELs ===
qrels = {}
with open("antique_qrels.txt", "r", encoding="utf-8") as f:
    for line in f:
        query_id, _, doc_id, relevance = line.strip().split()
        qrels.setdefault(query_id, {})[doc_id] = int(relevance)

# === Load TREC-style results ===
def load_run(file_path):
    run = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            run.setdefault(qid, {})[docid] = float(score)
    return run

# === Load each run ===
runs = {
    "TF-IDF": load_run("antique_tfidf_results.txt"),
    "BM25": load_run("antique_bm25_results.txt"),
    "BERT": load_run("antique_bert_results.txt"),
    "Hybrid": load_run("antique_hybrid_results.txt"),
}

# === Define metrics (pytrec_eval keys)
metrics = {'map', 'recip_rank', 'P_10', 'recall_1000'}

evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)

# === Evaluate and print results
for name, run in runs.items():
    results = evaluator.evaluate(run)

    avg_metrics = {
        metric: sum(query_scores[metric] for query_scores in results.values()) / len(results)
        for metric in metrics
    }

    print(f"\n📊 Results for {name}")
    print(f"  MAP:         {avg_metrics['map']:.4f}")
    print(f"  MRR:         {avg_metrics['recip_rank']:.4f}")
    print(f"  Recall@1000: {avg_metrics['recall_1000']:.4f}")
    print(f"  Precision@10:{avg_metrics['P_10']:.4f}")