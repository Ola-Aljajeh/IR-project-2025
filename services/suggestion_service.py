def get_suggestions(partial_query, dataset_name, clinical_queries, antique_queries, max_results=10):
    """
    Returns a list of suggested queries that match the user's input.
    """
    partial = partial_query.lower()

    if dataset_name == "clinical_trials":
        source = clinical_queries
    elif dataset_name == "antique":
        source = antique_queries
    else:
        return []

    matches = [q for q in source if partial in q.lower()]
    return matches[:max_results]
