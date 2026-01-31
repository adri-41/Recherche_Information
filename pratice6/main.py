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
# Extraire elements
# =======================
from bs4 import BeautifulSoup
import os

# -----------------------------
# Elements (tous les enfants d'article)
# -----------------------------
def load_collection_elements(root_dir):
    """
    Retourne une liste (docid, texte) pour chaque "element" dans les articles.
    Ici, on considère que chaque élément direct de <article> est un "element".
    """
    docs = []
    for fname in os.listdir(root_dir):
        if not fname.endswith(".xml"):
            continue
        path = os.path.join(root_dir, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "xml")
        article = soup.find("article")
        if article:
            # chaque enfant direct de <article> est un élément
            for i, elem in enumerate(article.find_all(recursive=False)):
                text = elem.get_text(" ", strip=True)
                if text:
                    docid = f"{os.path.splitext(fname)[0]}_elem{i+1}"
                    docs.append((docid, text))
    return docs


# -----------------------------
# Sections (<section> dans les articles)
# -----------------------------
def load_collection_sections(root_dir):
    """
    Retourne une liste (docid, texte) pour chaque <section> dans les articles.
    """
    docs = []
    for fname in os.listdir(root_dir):
        if not fname.endswith(".xml"):
            continue
        path = os.path.join(root_dir, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "xml")
        article = soup.find("article")
        if article:
            sections = article.find_all("section")
            for i, sec in enumerate(sections):
                text = sec.get_text(" ", strip=True)
                if text:
                    docid = f"{os.path.splitext(fname)[0]}_sec{i+1}"
                    docs.append((docid, text))
    return docs


# -----------------------------
# Paragraphes (<p> dans les articles)
# -----------------------------
def load_collection_paragraphs(root_dir):
    """
    Retourne une liste (docid, texte) pour chaque <p> dans les articles.
    """
    docs = []
    for fname in os.listdir(root_dir):
        if not fname.endswith(".xml"):
            continue
        path = os.path.join(root_dir, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "xml")
        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
            for i, p in enumerate(paragraphs):
                text = p.get_text(" ", strip=True)
                if text:
                    docid = f"{os.path.splitext(fname)[0]}_p{i+1}"
                    docs.append((docid, text))
    return docs


def load_collection_articles_fields(root_dir):

    stopset = set()  

    docs_fields = {}
    field_lens = {}
    df = defaultdict(int)


    total_len_field = defaultdict(int)
    count_docs = 0

    for fname in os.listdir(root_dir):
        if not fname.endswith(".xml"):
            continue

        docid = os.path.splitext(fname)[0]
        path = os.path.join(root_dir, fname)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "xml")

        article = soup.find("article")
        if not article:
            continue

        title_text = " ".join(t.get_text(" ", strip=True) for t in article.find_all("title"))
        section_text = " ".join(s.get_text(" ", strip=True) for s in article.find_all("section"))
        p_text = " ".join(p.get_text(" ", strip=True) for p in article.find_all("p"))
        rest_text = article.get_text(" ", strip=True)

        fields_raw = {
            "title": title_text,
            "section": section_text,
            "p": p_text,
            "rest": rest_text
        }

        fields_terms = {}
        fields_len = {}
        for field, txt in fields_raw.items():
            terms = tokenize_terms(txt) 
            fields_terms[field] = terms
            fields_len[field] = len(terms)
            total_len_field[field] += len(terms)

        docs_fields[docid] = fields_terms
        field_lens[docid] = fields_len

        seen = set()
        for field in fields_terms:
            seen.update(fields_terms[field])
        for t in seen:
            df[t] += 1

        count_docs += 1

    avg_field_len = {f: (total_len_field[f] / count_docs if count_docs else 0.0) for f in total_len_field}
    return docs_fields, field_lens, avg_field_len, df, count_docs


def score_query_bm25f(query, docs_fields, field_lens, avg_field_len, df, N,
                     field_weights=None, k1=1.2, b_field=None):

    if field_weights is None:
        field_weights = {"title": 3.0, "section": 1.5, "p": 1.0, "rest": 0.3}
    if b_field is None:
        b_field = {f: 0.75 for f in field_weights}

    q_terms = tokenize_terms(query)
    scores = defaultdict(float)

    for t in q_terms:
        if t not in df:
            continue

        # idf BM25 classique
        idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)

        for docid, fields in docs_fields.items():
            tf_prime = 0.0
            for f, w in field_weights.items():
                terms_f = fields.get(f, [])
                if not terms_f:
                    continue
                tf_f = 0

                for term in terms_f:
                    if term == t:
                        tf_f += 1
                if tf_f == 0:
                    continue

                len_f = field_lens[docid].get(f, 0)
                av_f = avg_field_len.get(f, 1.0) or 1.0
                bf = b_field.get(f, 0.75)
                norm = (1 - bf) + bf * (len_f / av_f)
                tf_prime += w * (tf_f / norm)

            if tf_prime > 0:
                scores[docid] += idf * ((k1 + 1) * tf_prime) / (k1 + tf_prime)

    return scores


# =======================
# MAIN
# =======================
def main():
    ensure_dir(OUTPUT_DIR)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Practice 6 – INEX\nTeam: {TEAM}\nVersion: v1\n\n")

    # ==========================
    # EXERCISE 1: Articles
    # ==========================
    print("=== Exercise 1: XML Articles ===")
    docs_articles = load_collection_xml(DATA_DIR)
    stop_full = load_stopwords(STOPFILE)
    N_articles = len(docs_articles)

    configs = [
        ("nostop", "nostem", set(), None),
        ("stop671", "nostem", stop_full, None),
        ("nostop", "porter", set(), PorterStemmer()),
        ("stop671", "porter", stop_full, PorterStemmer()),
    ]

    methods = ["tfidf", "lnu", "bm25"]
    stats_written_articles = False

    for stop_name, stem_name, stopset, stemmer in configs:
        postings, df, doc_len, stats = build_index(docs_articles, stopset, stemmer)
        avg_dl = sum(doc_len.values()) / N_articles

        if not stats_written_articles:
            write_report("=== Stats clés Articles ===")
            for k, v in stats.items():
                write_report(f"{k}: {v}")
            stats_written_articles = True

        for method in methods:
            run_name = f"run_articles_{stop_name}_{stem_name}_{method}.txt"
            run_path = os.path.join(OUTPUT_DIR, run_name)

            with open(run_path, "w", encoding="utf-8") as f:
                for qid, query in QUERIES.items():
                    scores = score_query(query, postings, df, doc_len, N_articles, method=method, avg_dl=avg_dl)
                    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]
                    for rank, (docid, score) in enumerate(ranked, 1):
                        f.write(f"{qid} Q0 {docid} {rank} {score:.4f} {TEAM}\n")

            print(f"[RUN OK] {run_name}")
            write_report(f"Run generated: {run_name}")

    # ==========================
    # EXERCISE 2: Elements
    # ==========================
    print("\n=== Exercise 2: XML Elements ===")
    docs_elements = load_collection_elements(DATA_DIR)  # À créer
    N_elements = len(docs_elements)
    stats_written_elements = False

    for stop_name, stem_name, stopset, stemmer in configs:
        postings, df, doc_len, stats = build_index(docs_elements, stopset, stemmer)
        avg_dl = sum(doc_len.values()) / N_elements

        if not stats_written_elements:
            write_report("=== Stats clés Elements ===")
            for k, v in stats.items():
                write_report(f"{k}: {v}")
            stats_written_elements = True

        for method in methods:
            # Runs BM25 avec tuning
            bm25_params = [(1.2, 0.75), (0.2, 0.25)] if method=="bm25" else [(None, None)]
            for k_val, b_val in bm25_params:
                run_name = f"run_elements_{stop_name}_{stem_name}_{method}"
                if method=="bm25":
                    run_name += f"_k{k_val}_b{b_val}"
                run_name += ".txt"
                run_path = os.path.join(OUTPUT_DIR, run_name)

                with open(run_path, "w", encoding="utf-8") as f:
                    for qid, query in QUERIES.items():
                        scores = score_query(query, postings, df, doc_len, N_elements, method=method, avg_dl=avg_dl, k=k_val, b=b_val)
                        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]
                        for rank, (docid, score) in enumerate(ranked, 1):
                            f.write(f"{qid} Q0 {docid} {rank} {score:.4f} {TEAM}\n")

                print(f"[RUN OK] {run_name}")
                write_report(f"Run generated: {run_name}")

    # ==========================
    # EXERCISE 2: Sections
    # ==========================
    print("\n=== Exercise 2: Sections ===")
    docs_sections = load_collection_sections(DATA_DIR)  # À créer
    N_sections = len(docs_sections)
    stats_written_sections = False

    for stop_name, stem_name, stopset, stemmer in configs:
        postings, df, doc_len, stats = build_index(docs_sections, stopset, stemmer)
        avg_dl = sum(doc_len.values()) / N_sections

        if not stats_written_sections:
            write_report("=== Stats clés Sections ===")
            for k, v in stats.items():
                write_report(f"{k}: {v}")
            stats_written_sections = True

        for method in methods:
            bm25_params = [(1.2, 0.75), (0.2, 0.25)] if method=="bm25" else [(None, None)]
            for k_val, b_val in bm25_params:
                run_name = f"run_sections_{stop_name}_{stem_name}_{method}"
                if method=="bm25":
                    run_name += f"_k{k_val}_b{b_val}"
                run_name += ".txt"
                run_path = os.path.join(OUTPUT_DIR, run_name)

                with open(run_path, "w", encoding="utf-8") as f:
                    for qid, query in QUERIES.items():
                        scores = score_query(query, postings, df, doc_len, N_sections, method=method, avg_dl=avg_dl, k=k_val, b=b_val)
                        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]
                        for rank, (docid, score) in enumerate(ranked, 1):
                            f.write(f"{qid} Q0 {docid} {rank} {score:.4f} {TEAM}\n")

                print(f"[RUN OK] {run_name}")
                write_report(f"Run generated: {run_name}")

    # ==========================
    # EXERCISE 2: Paragraphs
    # ==========================
    print("\n=== Exercise 2: Paragraphs ===")
    docs_paragraphs = load_collection_paragraphs(DATA_DIR)  # À créer
    N_paragraphs = len(docs_paragraphs)
    stats_written_paragraphs = False

    for stop_name, stem_name, stopset, stemmer in configs:
        postings, df, doc_len, stats = build_index(docs_paragraphs, stopset, stemmer)
        avg_dl = sum(doc_len.values()) / N_paragraphs

        if not stats_written_paragraphs:
            write_report("=== Stats clés Paragraphs ===")
            for k, v in stats.items():
                write_report(f"{k}: {v}")
            stats_written_paragraphs = True

        for method in methods:
            bm25_params = [(1.2, 0.75), (0.2, 0.25)] if method=="bm25" else [(None, None)]
            for k_val, b_val in bm25_params:
                run_name = f"run_paragraphs_{stop_name}_{stem_name}_{method}"
                if method=="bm25":
                    run_name += f"_k{k_val}_b{b_val}"
                run_name += ".txt"
                run_path = os.path.join(OUTPUT_DIR, run_name)

                with open(run_path, "w", encoding="utf-8") as f:
                    for qid, query in QUERIES.items():
                        scores = score_query(query, postings, df, doc_len, N_paragraphs, method=method, avg_dl=avg_dl, k=k_val, b=b_val)
                        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]
                        for rank, (docid, score) in enumerate(ranked, 1):
                            f.write(f"{qid} Q0 {docid} {rank} {score:.4f} {TEAM}\n")

                print(f"[RUN OK] {run_name}")
                write_report(f"Run generated: {run_name}")


    print("\n=== Exercise 3: XML Articles exploiting structure (BM25F) ===")

    docs_fields, field_lens, avg_field_len, df_struct, N_struct = load_collection_articles_fields(DATA_DIR)

    field_weights_list = [
        ("bm25f_w1", {"title": 3.0, "section": 1.5, "p": 1.0, "rest": 0.3}),
        ("bm25f_w2", {"title": 5.0, "section": 2.0, "p": 1.0, "rest": 0.2}),
    ]

    for name, fw in field_weights_list:
        run_name = f"run_articles_{name}.txt"
        run_path = os.path.join(OUTPUT_DIR, run_name)

        with open(run_path, "w", encoding="utf-8") as f:
            for qid, query in QUERIES.items():
                scores = score_query_bm25f(
                    query,
                    docs_fields, field_lens, avg_field_len,
                    df_struct, N_struct,
                    field_weights=fw,
                    k1=1.2
                )
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]
                for rank, (docid, score) in enumerate(ranked, 1):
                    f.write(f"{qid} Q0 {docid} {rank} {score:.4f} {TEAM}\n")

        print(f"[RUN OK] {run_name}")
        write_report(f"Run generated: {run_name}")


    # ==========================
    # ZIP
    # ==========================
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(REPORT_PATH)
        for f in os.listdir(OUTPUT_DIR):
            zipf.write(os.path.join(OUTPUT_DIR, f))

    print(f"\nArchive créée : {ZIP_NAME}")

if __name__ == "__main__":
    main()
