# src/boolean_ir.py
import os

def load_documents(path):
    """Return dict: filename -> list(tokens) from processed folder"""
    docs = {}
    if not os.path.exists(path):
        return docs
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".txt"):
            with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                docs[fname] = f.read().split()
    return docs

def build_inverted_index(docs):
    """Return dict term -> set(docnames)"""
    inverted = {}
    for doc_name, tokens in docs.items():
        for t in tokens:
            inverted.setdefault(t, set()).add(doc_name)
    return inverted

def parse_simple_boolean(query: str):
    """
    Parse very simple boolean queries:
    - "term1 and term2"
    - "term1 or term2"
    - "term1 not term2"
    - "term" single term
    Returns tuple (op, t1, t2) where op in {"AND","OR","NOT","TERM"}
    """
    q = query.lower().strip().split()
    if len(q) >= 3:
        if "and" in q:
            idx = q.index("and")
            return ("AND", q[idx-1], q[idx+1])
        if "or" in q:
            idx = q.index("or")
            return ("OR", q[idx-1], q[idx+1])
        if "not" in q:
            idx = q.index("not")
            return ("NOT", q[idx-1], q[idx+1])
    # fallback: first token as TERM
    return ("TERM", q[0] if q else "", None)

def search_boolean(query, inverted, all_docs):
    op, t1, t2 = parse_simple_boolean(query)
    if op == "AND":
        return inverted.get(t1, set()) & inverted.get(t2, set())
    if op == "OR":
        return inverted.get(t1, set()) | inverted.get(t2, set())
    if op == "NOT":
        return inverted.get(t1, set()) & (all_docs - inverted.get(t2, set()))
    return inverted.get(t1, set())
