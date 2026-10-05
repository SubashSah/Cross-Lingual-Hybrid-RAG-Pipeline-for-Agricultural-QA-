def normalize_scores(doc_scores, method="min_max"):
    """
    Normalizes a list of (document, score) tuples to the range [0, 1].

    Methods:
    - 'min_max': (score - min) / (max - min), scaling scores between 0 and 1.
    - 'max': score / max, keeping 0 at 0 and top score at 1.
    """
    if not doc_scores:
        return []

    scores = [s for _, s in doc_scores]
    min_s = min(scores)
    max_s = max(scores)

    normalized = []

    if method == "min_max":
        if max_s == min_s:
            return [(doc, 1.0 if max_s > 0 else 0.0) for doc, _ in doc_scores]
        for doc, score in doc_scores:
            norm_val = (score - min_s) / (max_s - min_s)
            normalized.append((doc, max(0.0, min(1.0, float(norm_val)))))
    elif method == "max":
        if max_s <= 0:
            return [(doc, 0.0) for doc, _ in doc_scores]
        for doc, score in doc_scores:
            norm_val = score / max_s
            normalized.append((doc, max(0.0, min(1.0, float(norm_val)))))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized
