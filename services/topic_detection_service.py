# services/topic_detection_service.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
import re

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return text

def detect_topics(clustered_results, n_topics=1, n_top_words=3):
    cluster_labels = {}

    for cluster_id, docs in clustered_results.items():
        texts = [preprocess_text(doc["text"]) for doc in docs]
        if not texts:
            cluster_labels[cluster_id] = ("", 0.0)
            continue

        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf = vectorizer.fit_transform(texts)

        try:
            nmf = NMF(n_components=n_topics, random_state=42, max_iter=1000)
            W = nmf.fit_transform(tfidf)
            H = nmf.components_
            feature_names = vectorizer.get_feature_names_out()

            topic_keywords = [feature_names[i] for i in H[0].argsort()[:-n_top_words - 1:-1]]
            topic_label = ", ".join(topic_keywords)
            confidence_score = float(np.mean(W[:, 0]))
            cluster_labels[cluster_id] = (topic_label, confidence_score)
        except Exception as e:
            cluster_labels[cluster_id] = ("", 0.0)

    return cluster_labels
