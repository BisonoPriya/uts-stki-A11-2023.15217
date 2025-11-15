# src/preprocess.py
import os
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load stemmer Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

STOPWORDS = set([
    "yang","dan","di","ke","dari","untuk","pada","dengan",
    "atau","juga","karena","ini","itu","dalam","adalah",
    "sebagai","dapat","telah","akan","agar","saja","itu","ini"
])

def clean(text: str) -> str:
    """Case folding + remove non-alphanumeric (keep numbers)"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str):
    """Split by whitespace"""
    return text.split()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]

def stem(tokens):
    return [stemmer.stem(t) for t in tokens]

def preprocess_text(text: str):
    """Full pipeline => list of tokens"""
    text = clean(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return tokens

def process_all(input_folder="data/raw", output_folder="data/processed"):
    """Process all .txt in input_folder and write processed tokens (space-separated) to output_folder"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".txt"):
            path = os.path.join(input_folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            tokens = preprocess_text(content)
            output_path = os.path.join(output_folder, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(" ".join(tokens))
            print(f"[OK] {filename} -> {output_path}")
