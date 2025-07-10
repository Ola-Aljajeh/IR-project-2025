from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import defaultdict
import numpy as np

def evaluate_clustering_quality(results, clustered_results, qrels, query, query_map):
    print("\n🧪 Sanity Check: Clustering Evaluation Inputs")

    # Attempt to map query text to query_id
    query_id = query_map.get(query)

    # Try fallback if no exact match
    if not query_id:
        print("⚠️  Exact query match not found. Attempting fallback...")
        for text, qid in query_map.items():
            if query.lower().strip() in text.lower().strip():
                query_id = qid
                print(f"✅ Fallback match: '{query}' → '{text}' (ID: {qid})")
                break

    if not query_id:
        print("❌ Could not find matching query ID. Skipping evaluation.")
        return {"purity": 0.0, "nmi": 0.0, "ari": 0.0}

    # Gather relevant doc IDs
    relevant_docs = qrels.get(query_id)
    if not relevant_docs:
        print("⚠️  No relevant documents found for query ID:", query_id)
        return {"purity": 0.0, "nmi": 0.0, "ari": 0.0}

    # Prepare y_true and y_pred
    y_true = []
    y_pred = []

    skipped = 0
    for cluster_id, docs in clustered_results.items():
        for doc in docs:
            doc_id = doc.get("id")
            if not doc_id or doc_id not in relevant_docs:
                skipped += 1
                continue
            y_true.append(relevant_docs[doc_id])
            y_pred.append(cluster_id)

    print(f"✅ Clustered results keys: {list(clustered_results.keys())}")
    print(f"📌 y_true sample: {y_true[:10]}")
    print(f"📌 y_pred sample: {y_pred[:10]}")
    print(f"⏭️ Skipped {skipped} docs without valid ID or relevance")

    if len(set(y_true)) <= 1:
        print("⚠️  Not enough relevant docs for evaluation.")
        return {"purity": 0.0, "nmi": 0.0, "ari": 0.0}

    # Purity calculation
    cluster_assignments = defaultdict(list)
    for label, pred in zip(y_true, y_pred):
        cluster_assignments[pred].append(label)

    total = len(y_true)
    purity_sum = sum(max(cluster.count(label) for label in set(cluster)) for cluster in cluster_assignments.values())
    purity = purity_sum / total
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    print("📊 Clustering Evaluation:")
    print(f"  Purity: {purity:.2f}")
    print(f"  NMI:    {nmi:.2f}")
    print(f"  ARI:    {ari:.2f}")

    return {"purity": purity, "nmi": nmi, "ari": ari}
