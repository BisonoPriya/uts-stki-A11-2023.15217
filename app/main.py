# app/main.py
import os
import sys
import math
import pprint

# color helper (optional)
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"

def line():
    print("\033[90m" + "─" * 70 + C.END)

try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    PROJECT_ROOT = os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.preprocess import process_all, preprocess_text
from src.boolean_ir import load_documents, build_inverted_index, search_boolean, parse_simple_boolean
from src.vsm import (
    load_docs,
    compute_tf_df_idf,
    build_tfidf_matrix_with_scheme,
    rank_documents_with_scheme,
)
from src.eval import (
    precision,
    recall,
    f1_score,
    precision_at_k,
    average_precision,
    ndcg_at_k,
)

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROC_PATH = os.path.join(PROJECT_ROOT, "data", "processed")

TRUTH_SETS = {
    "counter strike": {"doc4.txt", "doc1.txt", "doc8.txt"},
    "smoke flash": {"doc11.txt", "doc4.txt", "doc15.txt", "doc1.txt", "doc14.txt", "doc13.txt", "doc10.txt"},
    "awp recoil": {"doc11.txt", "doc15.txt", "doc14.txt"},
}

def run_preprocessing():
    line()
    print(f"{C.BOLD}{C.BLUE}=== STEP 1: PREPROCESSING ==={C.END}")
    line()
    if not os.path.exists(RAW_PATH):
        print(f"{C.RED}[ERROR]{C.END} Folder raw tidak ditemukan: {RAW_PATH}")
        return False
    raw_files = [f for f in os.listdir(RAW_PATH) if f.endswith('.txt')]
    if not raw_files:
        print(f"{C.YELLOW}[WARN]{C.END} Tidak ada file .txt di RAW folder")
        return False
    if not os.path.exists(PROC_PATH):
        os.makedirs(PROC_PATH)
    processed = [f for f in os.listdir(PROC_PATH) if f.endswith('.txt')]
    if not processed:
        print(f"{C.CYAN}Memproses dokumen...{C.END}")
        process_all(RAW_PATH, PROC_PATH)
    else:
        print(f"{C.GREEN}Dokumen sudah diproses sebelumnya.{C.END}")
    return True

def run_boolean_tests(docs, queries):
    line()
    print(f"{C.BOLD}{C.BLUE}=== STEP 2: BOOLEAN RETRIEVAL TESTS ==={C.END}")
    line()
    inv = build_inverted_index(docs)
    all_docs = set(docs.keys())
    results = {}
    for q in queries:
        res = search_boolean(q, inv, all_docs)
        results[q] = set(res)
        print(f"{C.CYAN}{C.BOLD}Query: {q}{C.END}")
        line()
        print(f"{C.YELLOW}Retrieved ({len(res)}):{C.END}")
        print(" ", ", ".join(sorted(res)))
        print(f"{C.GREEN}Gold relevan ({len(TRUTH_SETS[q])}):{C.END}")
        print(" ", ", ".join(sorted(TRUTH_SETS[q])))
        print(C.END)
    return results

def boolean_evaluation(results):
    line()
    print(f"{C.BOLD}{C.BLUE}=== STEP 3: EVALUASI BOOLEAN ==={C.END}")
    line()
    for q, retrieved in results.items():
        gold = TRUTH_SETS[q]
        p = precision(retrieved, gold)
        r = recall(retrieved, gold)
        f = f1_score(p, r)
        print(f"{C.CYAN}{C.BOLD}Query: {q}{C.END}")
        print(f"Precision: {p:.4f}  |  Recall: {r:.4f}  |  F1: {f:.4f}")
        line()

def run_vsm_tests(docs, queries, schemes=("standard", "sublinear")):
    line()
    print(f"{C.BOLD}{C.BLUE}=== STEP 4: VSM (TF-IDF) TESTS ==={C.END}")
    line()
    tf, df, idf = compute_tf_df_idf(docs)
    results = {}
    for scheme in schemes:
        print(f"{C.BOLD}{C.YELLOW}-- Skema: {scheme} --{C.END}")
        mat, doc_names, terms = build_tfidf_matrix_with_scheme(tf, idf, scheme=scheme)
        per_query = {}
        for q in queries:
            tokens = preprocess_text(q)
            ranked = rank_documents_with_scheme(tokens, tf, idf, mat, doc_names, terms, scheme)
            per_query[q] = ranked
            print(f"{C.CYAN}Query: {q}{C.END} | Skema: {scheme}")
            print(f"{'Rank':<6}{'Dokumen':<15}{'Score'}")
            for i, (doc, score) in enumerate(ranked[:5], start=1):
                print(f"{i:<6}{doc:<15}{score:.4f}")
            line()
        results[scheme] = per_query
    return results

def evaluate_vsm_rankings(rank_results):
    line()
    print(f"{C.BOLD}{C.BLUE}=== STEP 5: EVALUASI VSM ==={C.END}")
    line()
    k = 5
    summary = {}
    for scheme, perq in rank_results.items():
        print(f"{C.YELLOW}{C.BOLD}Scheme: {scheme}{C.END}")
        print(f"{'Query':<20}{'P@5':<10}{'MAP@5':<10}{'nDCG@5'}")
        agg = {"P@5": [], "MAP@5": [], "nDCG@5": []}
        for q, ranked in perq.items():
            gold = TRUTH_SETS[q]
            p5 = precision_at_k(ranked, gold, k)
            ap5 = average_precision(ranked, gold, k)
            nd5 = ndcg_at_k(ranked, gold, k)
            print(f"{q:<20}{p5:<10.4f}{ap5:<10.4f}{nd5:<10.4f}")
            agg["P@5"].append(p5); agg["MAP@5"].append(ap5); agg["nDCG@5"].append(nd5)
        avg = {m: (sum(v) / len(v) if v else 0.0) for m, v in agg.items()}
        print(f"RATA2 -> P@5={avg['P@5']:.4f}, MAP@5={avg['MAP@5']:.4f}, nDCG@5={avg['nDCG@5']:.4f}")
        line()
        summary[scheme] = avg
    return summary

def interactive_orchestrator(docs):
    line()
    print(f"{C.BOLD}{C.BLUE}=== STEP 6: SEARCH ENGINE ORCHESTRATOR ==={C.END}")
    line()
    inv = build_inverted_index(docs)
    tf, df, idf = compute_tf_df_idf(docs)
    while True:
        print("Pilih mode: \n1. Boolean\n2. VSM\n3. Exit")
        choice = input("Pilihan: ").strip()
        if choice == "3":
            print("Keluar orchestrator.")
            break
        q = input("Query: ").strip()
        if choice == "1":
            res = search_boolean(q, inv, set(docs.keys()))
            print("Hasil Boolean:", res)
        elif choice == "2":
            scheme = input("Skema (standard/sublinear): ").strip() or "standard"
            mat, doc_names, terms = build_tfidf_matrix_with_scheme(tf, idf, scheme)
            tokens = preprocess_text(q)
            ranked = rank_documents_with_scheme(tokens, tf, idf, mat, doc_names, terms, scheme)
            print(f"Top-5 VSM:")
            for d, s in ranked[:5]:
                print(f"{d} → {s:.4f}")

def chat_interface(docs):
    line()
    print(f"{C.BOLD}{C.BLUE}=== STEP 7: CHAT INTERFACE ==={C.END}")
    line()
    tf, df, idf = compute_tf_df_idf(docs)
    mat, doc_names, terms = build_tfidf_matrix_with_scheme(tf, idf, "sublinear")
    while True:
        q = input("Tulis pertanyaan/topik (atau 'exit'): ").strip()
        if q.lower() == "exit":
            break
        tokens = preprocess_text(q)
        ranked = rank_documents_with_scheme(tokens, tf, idf, mat, doc_names, terms, "sublinear")
        print(f"{C.GREEN}{C.BOLD}Ringkasan:{C.END}")
        for doc, score in ranked[:3]:
            row = mat[doc_names.index(doc)]
            top_terms = [terms[i] for i in list(reversed(row.argsort()))[:3] if row[i] > 0]
            print(f" - {doc} ({score:.3f}): {', '.join(top_terms)}")

def main():
    print(C.CYAN + C.BOLD)
    print("==============================================")
    print("       STKI SEARCH ENGINE ")
    print("==============================================" + C.END)

    ok = run_preprocessing()
    if not ok:
        return

    docs = load_docs(PROC_PATH)
    print(f"Memuat {len(docs)} dokumen...")

    queries = list(TRUTH_SETS.keys())

    bool_results = run_boolean_tests(docs, queries)
    boolean_evaluation(bool_results)

    vsm_rankings = run_vsm_tests(docs, queries, schemes=("standard", "sublinear"))
    vsm_summary = evaluate_vsm_rankings(vsm_rankings)

    print("(Mode interactive orchestrator — ketik '3' untuk keluar)")
    interactive_orchestrator(docs)

    chat_interface(docs)

    print(f"{C.GREEN}{C.BOLD}=== SEMUA SELESAI ==={C.END}")
    print("Ringkasan Skema TF-IDF:")
    pprint.pprint(vsm_summary)

if __name__ == "__main__":
    main()
