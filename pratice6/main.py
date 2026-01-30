import os
import re
import time
import math
import zipfile
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer

# =======================
# Config
# =======================
DATA_DIR = os.path.join(os.path.dirname(__file__), "Practice_05_data", "XML-Coll-withSem")
STOPFILE = os.path.join(os.path.dirname(__file__), "Practice_03_data", "stop-words-english4.txt")

OUTPUT_DIR = "generated_runs"
TEAM = "AdrienSoleneWilliam"

QUERIES = {
    "2009011": "olive oil health benefit",
    "2009036": "notting hill film actors",
    "2009067": "probabilistic models in information retrieval",
    "2009073": "web link network analysis",
    "2009074": "web ranking scoring algorithm",
    "2009078": "supervised machine learning algorithm",
    "2009085": "operating system mutual exclusion"
}

REPORT_PATH = "practice6v1_report.txt"
ZIP_NAME = f"practice6v1_{TEAM}.zip"

# =======================
# Utils
# =======================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def write_report(text):
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def load_stopwords(path):
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {w.strip().lower() for w in f if w.strip()}

# =======================
# Tokenisation
# =======================
def tokenize_tokens(text):
    return re.findall(r"[A-Za-zÀ-ÿ]+", text)

def tokenize_terms(text):
    return [t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ]+", text)]

def preprocess(tokens, stopset, stemmer, cache):
    out = []
    for t in tokens:
        if t in stopset:
            continue
        if stemmer:
            if t not in cache:
                cache[t] = stemmer.stem(t)
            out.append(cache[t])
        else:
            out.append(t)
    return out

# =======================
# Chargement XML
# =======================
def load_collection_xml(root_dir):
    docs = []
    for fname in os.listdir(root_dir):
        if not fname.endswith(".xml"):
            continue
        with open(os.path.join(root_dir, fname), encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "xml")
        article = soup.find("article")
        if article:
            docs.append((os.path.splitext(fname)[0], article.get_text(" ", strip=True)))
    return docs

# =======================
# Index
# =======================
def build_index(docs, stopset, stemmer):
    postings = defaultdict(lambda: defaultdict(int))
    df = defaultdict(int)
    doc_len = {}
    stem_cache = {}

    total_tokens = 0
    distinct_tokens = set()
    total_terms = 0
    distinct_terms = set()

    for docid, text in docs:
        tokens = tokenize_tokens(text)
        total_tokens += len(tokens)
        distinct_tokens.update(tokens)

        terms = preprocess(tokenize_terms(text), stopset, stemmer, stem_cache)
        doc_len[docid] = len(terms)
        total_terms += len(terms)
        distinct_terms.update(terms)

        tf = Counter(terms)
        for t, f in tf.items():
            postings[t][docid] = f
            df[t] += 1

    stats = {
        "total_tokens": total_tokens,
        "distinct_tokens": len(distinct_tokens),
        "avg_token_len": sum(len(t) for t in distinct_tokens) / len(distinct_tokens),
        "total_terms": total_terms,
        "distinct_terms": len(distinct_terms),
        "avg_doc_len": total_terms / len(docs),
        "avg_term_len": sum(len(t) for t in distinct_terms) / len(distinct_terms),
    }

    return postings, df, doc_len, stats

# =======================
# Scoring TF-IDF
# =======================
def score_query(query, postings, df, doc_len, N):
    scores = defaultdict(float)
    terms = tokenize_terms(query)
    for t in terms:
        if t not in postings:
            continue
        idf = math.log(N / df[t])
        for d, tf in postings[t].items():
            scores[d] += tf * idf
    return scores

# =======================
# Scoring TF-IDF simple, lnu et BM25
# =======================
def score_query(query, postings, df, doc_len, N, method="tfidf", avg_dl=None, k=1.2, b=0.75):
    scores = defaultdict(float)
    terms = tokenize_terms(query)
    for t in terms:
        if t not in postings:
            continue
        idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1) if method=="bm25" else math.log(N / df[t])
        for d, tf in postings[t].items():
            if method == "tfidf":
                scores[d] += tf * idf
            elif method == "lnu":
                scores[d] += (tf / doc_len[d]) * idf
            elif method == "bm25":
                denom = tf + k * (1 - b + b * (doc_len[d] / avg_dl))
                scores[d] += idf * tf * (k + 1) / denom
    return scores

# =======================
# MAIN
# =======================
def main():
    ensure_dir(OUTPUT_DIR)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Practice 6 – INEX\nTeam: {TEAM}\nVersion: v1\n\n")

    docs = load_collection_xml(DATA_DIR)
    stop_full = load_stopwords(STOPFILE)
    N = len(docs)

    # Configurations stop/stem
    configs = [
        ("nostop", "nostem", set(), None),
        ("stop671", "nostem", stop_full, None),
        ("nostop", "porter", set(), PorterStemmer()),
        ("stop671", "porter", stop_full, PorterStemmer()),
    ]

    # Méthodes de scoring
    methods = ["tfidf", "lnu", "bm25"]

    # Paramètres BM25 supplémentaires
    bm25_tuning = [
        (1.2, 0.75),  # valeur par défaut
        (0.2, 0.25),  # tuning demandé
    ]

    stats_written = False

    for stop_name, stem_name, stopset, stemmer in configs:
        print(f"\n--- Run stop={stop_name}, stem={stem_name} ---")
        postings, df, doc_len, stats = build_index(docs, stopset, stemmer)
        avg_dl = sum(doc_len.values()) / N  # moyenne longueur doc pour BM25/lnu

        # -----------------------
        # Affichage et report stats (une seule fois)
        # -----------------------
        if not stats_written:
            write_report("=== Stats clés (tous runs) ===")
            print("\n=== Stats clés (tous runs) ===")
            for k, v in stats.items():
                line = f"{k}: {v}"
                print(line)
                write_report(line)
            stats_written = True

        # -----------------------
        # Génération des runs
        # -----------------------
        for method in methods:
            if method == "bm25":
                # Pour BM25, on fait tous les tunings
                for k_val, b_val in bm25_tuning:
                    run_name = f"run_articles_{stop_name}_{stem_name}_{method}_k{k_val}_b{b_val}.txt"
                    run_path = os.path.join(OUTPUT_DIR, run_name)

                    with open(run_path, "w", encoding="utf-8") as f:
                        for qid, query in QUERIES.items():
                            scores = score_query(query, postings, df, doc_len, N,
                                                 method=method, avg_dl=avg_dl, k=k_val, b=b_val)
                            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]
                            for rank, (docid, score) in enumerate(ranked, 1):
                                f.write(f"{qid} Q0 {docid} {rank} {score:.4f} {TEAM}\n")

                    print(f"[RUN OK] {run_name}")
                    write_report(f"Run generated: {run_name}")
            else:
                # tfidf ou lnu
                run_name = f"run_articles_{stop_name}_{stem_name}_{method}.txt"
                run_path = os.path.join(OUTPUT_DIR, run_name)

                with open(run_path, "w", encoding="utf-8") as f:
                    for qid, query in QUERIES.items():
                        scores = score_query(query, postings, df, doc_len, N,
                                             method=method, avg_dl=avg_dl)
                        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]
                        for rank, (docid, score) in enumerate(ranked, 1):
                            f.write(f"{qid} Q0 {docid} {rank} {score:.4f} {TEAM}\n")

                print(f"[RUN OK] {run_name}")
                write_report(f"Run generated: {run_name}")

    # -----------------------
    # Création archive ZIP
    # -----------------------
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(REPORT_PATH)
        for f in os.listdir(OUTPUT_DIR):
            zipf.write(os.path.join(OUTPUT_DIR, f))
        # ajoute d'autres .py si besoin sauf main.py
        for f in os.listdir("."):
            if f.endswith(".py") and f != os.path.basename(__file__):
                zipf.write(f)

    print(f"\nArchive créée : {ZIP_NAME}")

if __name__ == "__main__":
    main()
