import os
import re
import time
import math
import zipfile
from collections import defaultdict, Counter

# Essayez d'importer NLTK PorterStemmer ; si absent on propose DummyStemmer
try:
    from nltk.stem import PorterStemmer
except Exception:
    PorterStemmer = None


# ---------------------------
# Config / paramètres
# ---------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "Practice_03_data")
DATAFILE = os.path.join(DATA_DIR, "Text_Only_Ascii_Coll_NoSem")
STOPFILE = os.path.join(DATA_DIR, "stop-words-english4.txt")
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
# QUERIES = {"1":"web ranking scoring algorithm"}

TOP_K = 1500

# BM25 default params
BM25_K1 = 1.2
BM25_B = 0.75

TOKEN_RE = re.compile(r"[a-z]+")  # tokenizer strict (minuscule)


# ---------------------------
# Utilitaires
# ---------------------------
class DummyStemmer:
    def stem(self, t):
        return t


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ---------------------------
# I/O : chargement collection & stopwords
# ---------------------------
DOC_PATTERN = re.compile(r"<doc>\s*<docno>\s*([^<\s]+)\s*</docno>(.*?)</doc>",
                         flags=re.IGNORECASE | re.DOTALL)


def load_collection(path):
    """Lecture de la collection entière et renvoi d'une liste (docid, content)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Collection introuvable : {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    docs = []
    for m in DOC_PATTERN.finditer(text):
        docs.append((m.group(1).strip(), m.group(2)))
    return docs


def load_stopwords(path):
    """Charge la liste de stop-words (lowercase)."""
    if not path or not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return set(line.strip().lower() for line in f if line.strip())


# ---------------------------
# Tokenisation / prétraitement
# ---------------------------
def tokenizer(text):
    """Renvoie la liste de tokens normalisés (minuscules)."""
    return TOKEN_RE.findall(text.lower())


def preprocess_tokens(tokens, stopset, stemmer, stem_cache):
    """Supprime stopwords et applique le stemming via stemmer (utilise cache)."""
    out = []
    for t in tokens:
        if t in stopset:
            continue
        if stemmer is None:
            out.append(t)
        else:
            # cache du stem pour gagner du temps
            s = stem_cache.get(t)
            if s is None:
                s = stemmer.stem(t)
                stem_cache[t] = s
            out.append(s)
    return out


# ---------------------------
# Construction d'index
# ---------------------------
def build_index(docs, stopset, stemmer):
    """
    Construit postings, df, doc_len (length after preprocessing).
    - docs : liste (docid, content)
    - stopset : set de stopwords (lowercase)
    - stemmer : instance ayant .stem() ou None
    Retour : postings (term -> {docid: tf}), df (term -> docfreq), doc_len (docid -> len),
             doc_ids (liste), stem_cache (dict)
    """
    stem_cache = {}
    postings = defaultdict(lambda: defaultdict(int))
    df = defaultdict(int)
    doc_len = {}
    doc_ids = []

    for docid, content in docs:
        doc_ids.append(docid)
        tokens = tokenizer(content)
        terms = preprocess_tokens(tokens, stopset, stemmer, stem_cache)
        doc_len[docid] = len(terms)
        tf_counter = Counter(terms)
        for term, tf in tf_counter.items():
            postings[term][docid] = tf
            df[term] += 1

    return postings, df, doc_len, doc_ids, stem_cache


# ---------------------------
# LTN : calcul des poids et scoring
# ---------------------------
def compute_ltn_weights(postings, df, N):
    """
    Compute weighted_postings: term -> {docid: w_td}
    w_td = (1 + log10(tf_td)) * log10(N/df_t)
    """
    weighted = {}
    idf = {}
    for t, df_t in df.items():
        if df_t > 0:
            idf[t] = math.log10(N / df_t)
        else:
            idf[t] = 0.0

    for t, plist in postings.items():
        idf_t = idf.get(t, 0.0)
        if idf_t <= 0.0:
            weighted[t] = {}
            continue
        wmap = {}
        for d, tf in plist.items():
            if tf > 0:
                wmap[d] = (1.0 + math.log10(tf)) * idf_t
        weighted[t] = wmap
    return weighted, idf


def score_query_ltn(weighted_postings, query_terms):
    """
    Scoring LTN with query weighting (1 + log10(tf_q)) * (sum w_td * w_tq)
    Implementation mirrors your previous practice where query is weighted.
    """
    q_tf = Counter(query_terms)
    q_w = {t: (1.0 + math.log10(tf)) for t, tf in q_tf.items() if tf > 0}
    scores = defaultdict(float)
    for t, wq in q_w.items():
        postings_t = weighted_postings.get(t, {})
        for d, wtd in postings_t.items():
            scores[d] += wtd * wq
    return scores


# ---------------------------
# LTC : compute weights with doc normalization and scoring (lnn query)
# ---------------------------
def compute_ltc_weights(postings, df, N):
    """
    Compute weighted_postings normalized per document (ltc):
    w_td = (1 + log10(tf_td)) * idf_t
    then divide by doc norm (sqrt(sum w_td^2)) per doc
    Returns weighted_postings (term -> {doc: w_td_norm}) and doc_norms (raw norms)
    """
    weighted = {}
    doc_norm_sq = defaultdict(float)

    # first pass: compute raw w_td and accumulate norm squared
    for t, plist in postings.items():
        df_t = df.get(t, 0)
        if df_t <= 0:
            continue
        idf_t = math.log10(N / df_t)
        for d, tf in plist.items():
            if tf <= 0:
                continue
            w = (1.0 + math.log10(tf)) * idf_t
            weighted.setdefault(t, {})[d] = w
            doc_norm_sq[d] += w * w

    # second pass: normalize per document
    for t, plist in weighted.items():
        for d, w in list(plist.items()):
            norm = math.sqrt(doc_norm_sq.get(d, 1.0))
            if norm > 0:
                plist[d] = w / norm
            else:
                plist[d] = 0.0

    return weighted, doc_norm_sq


def score_query_ltc(weighted_postings, query_terms):
    """
    Query weighting using lnn for the query: w_tq = 1 + log10(tf_q)
    Score is dot product of normalized doc vectors and query weights
    """
    q_tf = Counter(query_terms)
    q_w = {t: 1.0 + math.log10(tf) for t, tf in q_tf.items() if tf > 0}
    scores = defaultdict(float)
    for t, wq in q_w.items():
        plist = weighted_postings.get(t)
        if not plist:
            continue
        for d, wtd in plist.items():
            scores[d] += wtd * wq
    return scores


# ---------------------------
# BM25
# ---------------------------
def score_query_bm25(postings, df, doc_len, N, query_terms, k1=BM25_K1, b=BM25_B):
    """
    Standard BM25 scoring (idf using log((N-df+0.5)/(df+0.5))).
    Returns dict(docid -> score), avdl
    """
    if len(doc_len) == 0:
        return {}, 0.0
    avdl = sum(doc_len.values()) / len(doc_len)
    idf = {}
    for t, df_t in df.items():
        if df_t > 0:
            idf[t] = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1e-12)  # small eps
        else:
            idf[t] = 0.0

    scores = defaultdict(float)
    q_terms_set = set(query_terms)
    for t in q_terms_set:
        if t not in postings:
            continue
        idf_t = idf.get(t, 0.0)
        for d, tf in postings[t].items():
            dl = doc_len.get(d, avdl)
            denom = tf + k1 * ((1.0 - b) + b * (dl / avdl))
            tf_adj = (tf * (k1 + 1.0)) / denom
            scores[d] += idf_t * tf_adj
    return scores, avdl


# ---------------------------
# Helper pour top-k + padding
# ---------------------------
def top_k_with_padding(scores_dict, all_doc_ids, k=TOP_K):
    """
    Returns top-k list of (docid, score). If less than k docs have non-zero scores,
    pad with docids (score=0.0) from all_doc_ids (in deterministic order).
    """
    ranked = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) >= k:
        return ranked[:k]
    # pad
    used = set(d for d, _ in ranked)
    pad = []
    for d in all_doc_ids:
        if d not in used:
            pad.append((d, 0.0))
            if len(ranked) + len(pad) >= k:
                break
    return ranked + pad


# ---------------------------
# Run generation (single combo)
# ---------------------------
def generate_one_run(run_name, method, postings, df, doc_len, doc_ids, N, queries,
                     stopset, stemmer, stem_cache, out_dir):
    """
    method in {'ltn','ltc','bm25'}
    """
    ensure_dir(out_dir)
    run_path = os.path.join(out_dir, f"{TEAM}_{run_name}_{method}.txt")
    # Precompute weights if needed
    weighted = None
    extra = {}
    if method == "ltn":
        weighted, _ = compute_ltn_weights(postings, df, N)
    elif method == "ltc":
        weighted, _ = compute_ltc_weights(postings, df, N)
    # bm25 doesn't need pre-weight

    lines_written = 0
    with open(run_path, "w", encoding="utf-8") as f:
        for qid, qtext in queries.items():
            q_tokens_raw = tokenizer(qtext)
            q_terms = preprocess_tokens(q_tokens_raw, stopset, stemmer, stem_cache)
            # score according to method
            try:
                if method == "ltn":
                    scores = score_query_ltn(weighted, q_terms)
                elif method == "ltc":
                    scores = score_query_ltc(weighted, q_terms)
                elif method == "bm25":
                    scores, _ = score_query_bm25(postings, df, doc_len, N, q_terms)
                else:
                    scores = {}
            except Exception as e:
                print(f"[WARN] erreur scoring {method} q={qid} : {e}")
                scores = {}

            topk = top_k_with_padding(scores, doc_ids, TOP_K)
            for rank, (docid, score) in enumerate(topk, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.5f} {TEAM} /article[1]\n")
                lines_written += 1

    expected = len(queries) * TOP_K
    return run_path, lines_written, expected


# ---------------------------
# Main: génération des 12 runs
# ---------------------------
def main():
    print("=== Génération des 12 runs (nostop/stop × nostem/porter × ltn/ltc/bm25) ===")
    # Vérifications
    if not os.path.exists(DATAFILE):
        print(f"[ERROR] Collection manquante : {DATAFILE}")
        return
    # load collection once
    print("Lecture de la collection (une seule fois)...")
    docs = load_collection(DATAFILE)
    print(f"Documents chargés : {len(docs)}")

    # load stopwords set
    stop_full = load_stopwords(STOPFILE)

    stop_options = [("nostop", set()), ("stop671", stop_full)]
    stem_options = [("nostem", None), ("porter", PorterStemmer() if PorterStemmer else None)]

    methods = ["ltn", "ltc", "bm25"]

    ensure_dir(OUTPUT_DIR)
    run_paths = []

    run_id = 1
    for stop_name, stopset in stop_options:
        for stem_name, stemmer in stem_options:
            # create stem_cache (shared per combination)
            stem_cache = {}
            # build index once for this (stop, stem) combo
            print(f"\n--- Construction index (stop={stop_name}, stem={stem_name}) ---")
            t0 = time.time()
            postings, df, doc_len, doc_ids, stem_cache = build_index(docs, stopset, stemmer)
            N = len(doc_ids)
            t_index = time.time() - t0
            print(f"Index construit: terms={len(df):,}, docs={N:,} (temps {t_index:.2f}s)")

            for method in methods:
                run_name = f"{run_id}_{method}_article_{stop_name}_{stem_name}"
                print(f"→ Génération run {run_name} ...")
                t0 = time.time()
                path, written, expected = generate_one_run(
                    run_name, method, postings, df, doc_len, doc_ids, N,
                    QUERIES, stopset, stemmer, stem_cache, OUTPUT_DIR
                )
                elapsed = time.time() - t0
                run_paths.append(path)
                ok = "OK" if written == expected else f"INCOMPLET ({written}/{expected})"
                print(f"   -> {os.path.basename(path)}  (lignes {written}/{expected})  time scoring: {elapsed:.2f}s  {ok}")
                run_id += 1

    # pack zip
    zipname = f"{TEAM}_ALL_RUNS.zip"
    zip_path = os.path.join(OUTPUT_DIR, zipname)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in run_paths:
            zf.write(p, os.path.basename(p))
    print(f"\nZIP généré : {zip_path}")
    print("Terminé.")


if __name__ == "__main__":
    main()
