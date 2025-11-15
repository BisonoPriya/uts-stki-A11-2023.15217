from .preprocess import preprocess_text
from .boolean_ir import load_documents, build_inverted_index, search_boolean
from .vsm import load_docs, compute_tf_df_idf, build_tfidf_matrix_with_scheme, rank_documents_with_scheme

class SearchEngine:
    def __init__(self, data_path="data/processed"):
        self.data_path = data_path
        self.docs = load_docs(self.data_path)
        self.inv = build_inverted_index(self.docs)
        self.tf, self.df, self.idf = compute_tf_df_idf(self.docs)

    def search(self, query, model="vsm", scheme="standard", k=5):
        query_tokens = preprocess_text(query)
        if model == "boolean":
            all_docs = set(self.docs.keys())
            result = search_boolean(query, self.inv, all_docs)
            return list(result)
        elif model == "vsm":
            mat, doc_names, terms = build_tfidf_matrix_with_scheme(self.tf, self.idf, scheme)
            ranked = rank_documents_with_scheme(query_tokens, self.tf, self.idf, mat, doc_names, terms, scheme)
            return ranked[:k]
        else:
            raise ValueError("Unknown model")