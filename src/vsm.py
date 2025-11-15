import os
import math
from collections import Counter
import numpy as np

def load_docs(path):
    """Return dict: filename -> list(tokens)"""
    docs = {}
    if not os.path.exists(path):
        return docs
    for f in sorted(os.listdir(path)):
        if f.endswith(".txt"):
            with open(os.path.join(path, f), "r", encoding="utf-8") as x:
                docs[f] = x.read().split()
    return docs

def compute_tf_df_idf(docs):
    """Return tf(dict(doc->Counter)), df(Counter), idf(dict(term->float))"""
    tf = {}
    df = Counter()
    for doc, tokens in docs.items():
        tf[doc] = Counter(tokens)
        for t in set(tokens):
            df[t] += 1
    N = max(1, len(docs))
    idf = {t: math.log10(N / df[t]) if df[t] > 0 else 0.0 for t in df}
    return tf, df, idf

def build_tfidf_matrix_with_scheme(tf, idf, scheme="standard"):
    """
    scheme: 'standard' => tf * idf
            'sublinear' => log(1+tf) * idf
    Returns: matrix (N_docs x M_terms), doc_names(list), terms(list)
    """
    terms = sorted(idf.keys())
    index = {t: i for i, t in enumerate(terms)}
    docs = list(tf.keys())
    mat = np.zeros((len(docs), len(terms)), dtype=float)
    for i, doc in enumerate(docs):
        for term, freq in tf[doc].items():
            if term not in index:
                continue
            if scheme == "standard":
                w = freq * idf[term]
            elif scheme == "sublinear":
                w = math.log1p(freq) * idf[term]
            else:
                raise ValueError("Unknown scheme")
            mat[i, index[term]] = w
    return mat, docs, terms

def rank_documents_with_scheme(query_tokens, tf, idf, mat, docs, terms, scheme="standard"):
    q_tf = Counter(query_tokens)
    q_vec = np.zeros(len(terms), dtype=float)
    for t, f in q_tf.items():
        if t in idf and t in terms:
            if scheme == "standard":
                q_vec[terms.index(t)] = f * idf[t]
            else:
                q_vec[terms.index(t)] = math.log1p(f) * idf[t]
    def cos(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return 0.0 if denom == 0 else float(np.dot(a, b) / denom)
    scores = [(docs[i], cos(q_vec, mat[i])) for i in range(len(docs))]
    return sorted(scores, key=lambda x: x[1], reverse=True)