import numpy as np

def precision(retrieved_set, relevant_set):
    if len(retrieved_set) == 0:
        return 0.0
    return len(set(retrieved_set) & set(relevant_set)) / len(set(retrieved_set))

def recall(retrieved_set, relevant_set):
    if len(relevant_set) == 0:
        return 0.0
    return len(set(retrieved_set) & set(relevant_set)) / len(set(relevant_set))

def f1_score(prec, rec):
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

def precision_at_k(ranked_list, relevant_docs, k=5):
    retrieved_k = [doc for doc, _ in ranked_list[:k]]
    hits = [1 if doc in relevant_docs else 0 for doc in retrieved_k]
    return np.sum(hits) / k

def average_precision(ranked_list, relevant_docs, k=None):
    if k:
        ranked_list = ranked_list[:k]
    precisions = []
    hit_count = 0
    for i, (doc, _) in enumerate(ranked_list, start=1):
        if doc in relevant_docs:
            hit_count += 1
            precisions.append(hit_count / i)
    return float(np.mean(precisions)) if precisions else 0.0

def ndcg_at_k(ranked_list, relevant_docs, k=5):
    def dcg(scores):
        return np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(scores)])
    ranked = ranked_list[:k]
    scores = [1 if doc in relevant_docs else 0 for doc, _ in ranked]
    ideal = sorted(scores, reverse=True)
    dcg_val = dcg(scores)
    idcg_val = dcg(ideal)
    return float(dcg_val / idcg_val) if idcg_val > 0 else 0.0