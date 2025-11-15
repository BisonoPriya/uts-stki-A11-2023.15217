1. Deskripsi Proyek

Proyek ini mengimplementasikan mini search engine berbasis dua model temu kembali informasi:

Boolean Retrieval (AND, OR, NOT)

Vector Space Model (VSM) dengan:

TF–IDF Standard

TF–IDF Sublinear (log normalization)

Pipeline sistem mencakup preprocessing → indexing → retrieval → ranking → evaluasi → mode interaktif. Seluruh modul bersifat modular sehingga dapat dijalankan maupun diuji secara terpisah.

Cara Menjalankan
A. Persyaratan

Python 3.9 – 3.12

pip

B. Instalasi
pip install -r requirements.txt

C. Menjalankan Sistem
python app/main.py


Program akan menjalankan otomatis:

Preprocessing

Boolean Retrieval

Evaluasi Boolean

VSM (TF-IDF Standard & Sublinear)

Evaluasi VSM

Interactive Search Mode

Chat Summarizer Mode

 Fitur Utama
1) Preprocessing

Lowercasing

Tokenisasi

Stopword removal

Stemming sederhana
→ Output tersimpan di data/processed/

2) Boolean Retrieval

Operator: AND, OR, NOT

Menggunakan inverted index

3) Vector Space Model

Pembobotan:

TF–IDF Standard

TF–IDF Sublinear

Perankingan: Cosine Similarity

4) Evaluasi IR

Precision

Recall

F1

Precision@5

MAP@5

nDCG@5


Interactive Mode

Mendukung dua model:

Boolean

VSM



Chat Summarizer

Menghasilkan ringkasan otomatis berdasarkan top-3 dokumen VSM.
