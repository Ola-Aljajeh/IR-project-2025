# cluster_service.py
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

def cluster_documents(results, k):
    """
    Cluster the documents using cosine distance on their embeddings.
    Returns a dictionary: {cluster_id: [docs...]}
    """
    if not results or k <= 1:
        return {0: results}

    # Extract embeddings
    embeddings = [doc["embedding"] for doc in results if "embedding" in doc]
    if not embeddings:
        return {0: results}  # fallback to no clustering

    X = np.vstack(embeddings)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    clustered = {i: [] for i in range(k)}
    for doc, label in zip(results, labels):
        clustered[label].append(doc)

    return clustered
