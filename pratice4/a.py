import os
import re
import time
import math
import zipfile
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer


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

TOP_K = 1500

# BM25 default params
BM25_K1 = 1.2
BM25_B = 0.75

TOKEN_RE = re.compile(r"[A-Za-z]+")  # pour tokens (stats)
TERM_RE = re.compile(r"[a-z]+")  # pour terms (indexation)


def tokenizer_tokens(text):
    return TOKEN_RE.findall(text)


def tokenizer_terms(text):
    return TERM_RE.findall(text.lower())


# Analyse single doc/term:
TARGET_DOC = "23724"
TARGET_QUERY_ID = "2009074"
TARGET_QUERY_TEXT = QUERIES[TARGET_QUERY_ID]
TARGET_TERM = "ranking"


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
DOC_PATTERN = re.compile(
    r"<doc>\s*<docno>\s*([^<\s]+)\s*</docno>(.*?)</doc>",
    flags=re.IGNORECASE | re.DOTALL
)


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
    Construit l'index inversé, calcule DF, doc_len, stem_cache et les statistiques.
    """
    stem_cache = {}
    postings = defaultdict(lambda: defaultdict(int))
    df = defaultdict(int)
    doc_len = {}
    doc_ids = []

    # Stats
    total_tokens = 0
    total_token_chars = 0
    distinct_tokens = set()
    total_terms = 0
    total_term_chars = 0
    distinct_terms = set()

    for docid, content in docs:
        doc_ids.append(docid)

        # --- Tokens pour stats ---
        tokens = re.findall(r"[A-Za-z]+", content)
        total_tokens += len(tokens)
        distinct_tokens.update(tokens)
        total_token_chars += sum(len(t) for t in tokens)

        # --- Terms pour index ---
        terms = re.findall(r"[a-z]+", content.lower())
        terms = preprocess_tokens(terms, stopset, stemmer, stem_cache)

        doc_len[docid] = len(terms)
        total_terms += len(terms)
        total_term_chars += sum(len(t) for t in terms)
        distinct_terms.update(terms)

        # Construction de l'index inversé
        tf_counter = Counter(terms)
        for term, tf in tf_counter.items():
            postings[term][docid] = tf
            df[term] += 1

    stats = {
        "total_tokens": total_tokens,
        "distinct_tokens": len(distinct_tokens),
        "avg_token_len": sum(len(t) for t in distinct_tokens) / len(distinct_tokens) if distinct_tokens else 0,
        "total_terms": total_terms,
        "distinct_terms": len(distinct_terms),
        "avg_doc_len": total_terms / len(docs) if docs else 0,
        "avg_term_len": sum(len(t) for t in distinct_terms) / len(distinct_terms) if distinct_terms else 0,
    }

    return postings, df, doc_len, doc_ids, stem_cache, stats


# ---------------------------
# LTN
# ---------------------------
def compute_ltn_weights(postings, df, N):
    weighted = {}
    idf = {}
    for t, df_t in df.items():
        idf[t] = math.log10(N / df_t) if df_t > 0 else 0.0

    for t, plist in postings.items():
        idf_t = idf[t]
        if idf_t <= 0:
            weighted[t] = {}
            continue
        wmap = {}
        for d, tf in plist.items():
            wmap[d] = (1 + math.log10(tf)) * idf_t
        weighted[t] = wmap
    return weighted, idf


def score_query_ltn(weighted_postings, query_terms):
    q_tf = Counter(query_terms)
    q_w = {t: 1 + math.log10(tf) for t, tf in q_tf.items()}
    scores = defaultdict(float)

    for t, wq in q_w.items():
        for d, wtd in weighted_postings.get(t, {}).items():
            scores[d] += wtd * wq

    return scores


# ---------------------------
# LTC
# ---------------------------
def compute_ltc_weights(postings, df, N):
    weighted = {}
    doc_norm_sq = defaultdict(float)

    # compute raw weights
    for t, plist in postings.items():
        df_t = df[t]
        if df_t <= 0:
            continue
        idf_t = math.log10(N / df_t)
        for d, tf in plist.items():
            w = (1 + math.log10(tf)) * idf_t
            weighted.setdefault(t, {})[d] = w
            doc_norm_sq[d] += w * w

    # normalize
    for t, plist in weighted.items():
        for d, w in list(plist.items()):
            norm = math.sqrt(doc_norm_sq[d])
            plist[d] = w / norm if norm > 0 else 0.0

    return weighted, doc_norm_sq


def score_query_ltc(weighted_postings, query_terms):
    q_tf = Counter(query_terms)
    q_w = {t: 1 + math.log10(tf) for t, tf in q_tf.items()}
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
    if not doc_len:
        return {}, 0
    avdl = sum(doc_len.values()) / len(doc_len)

    scores = defaultdict(float)

    for t in set(query_terms):
        if t not in postings:
            continue

        df_t = df[t]
        idf_t = math.log((N - df_t + 0.5) / (df_t + 0.5))

        for d, tf in postings[t].items():
            dl = doc_len[d]
            denom = tf + k1 * ((1 - b) + b * (dl / avdl))
            tf_adj = (tf * (k1 + 1)) / denom
            scores[d] += idf_t * tf_adj

    return scores, avdl


# ---------------------------
# Helper top-k avec padding
# ---------------------------
def top_k_with_padding(scores, doc_ids, k=TOP_K):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) >= k:
        return ranked[:k]

    used = set(d for d, _ in ranked)
    pad = [(d, 0.0) for d in doc_ids if d not in used][:k - len(ranked)]
    return ranked + pad


# ---------------------------
# Analyse single document + top5
# ---------------------------
def analyse_single_method(method, postings, df, doc_len, N,
                          stopset, stemmer, stem_cache):
    print(f"\n========== ANALYSE {method.upper()} (doc {TARGET_DOC}) ==========")

    q_terms = preprocess_tokens(
        tokenizer(TARGET_QUERY_TEXT),
        stopset, stemmer, stem_cache
    )

    tok = TARGET_TERM.lower()
    term_stem = tok if stemmer is None else stemmer.stem(tok)

    # ---- LTN ----
    if method == "ltn":
        weighted, _ = compute_ltn_weights(postings, df, N)
        scores = score_query_ltn(weighted, q_terms)

        w_td = weighted.get(term_stem, {}).get(TARGET_DOC, 0)
        print(f"Weight('{TARGET_TERM}', {TARGET_DOC}) = {w_td:.6f}")
        print(f"RSV(q,{TARGET_DOC}) = {scores.get(TARGET_DOC, 0):.6f}")

        top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop-5 :")
        for i, (d, s) in enumerate(top5, 1):
            print(f"{i}. {d}  {s:.6f}")

    # ---- LTC ----
    elif method == "ltc":
        weighted, _ = compute_ltc_weights(postings, df, N)
        scores = score_query_ltc(weighted, q_terms)

        w_td = weighted.get(term_stem, {}).get(TARGET_DOC, 0)
        print(f"Weight('{TARGET_TERM}', {TARGET_DOC}) = {w_td:.6f}")
        print(f"RSV(q,{TARGET_DOC}) = {scores.get(TARGET_DOC, 0):.6f}")

        top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop-5 :")
        for i, (d, s) in enumerate(top5, 1):
            print(f"{i}. {d}  {s:.6f}")

    # ---- BM25 ----
    else:
        scores, avdl = score_query_bm25(postings, df, doc_len, N, q_terms)

        tf = postings.get(term_stem, {}).get(TARGET_DOC, 0)
        if tf > 0:
            dl = doc_len[TARGET_DOC]
            denom = tf + BM25_K1 * ((1 - BM25_B) + BM25_B * (dl / avdl))
            tf_adj = (tf * (BM25_K1 + 1)) / denom
        else:
            tf_adj = 0.0

        print(f"Weight('{TARGET_TERM}', {TARGET_DOC}) = {tf_adj:.6f}")
        print(f"RSV(q,{TARGET_DOC}) = {scores.get(TARGET_DOC, 0):.6f}")

        top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop-5 :")
        for i, (d, s) in enumerate(top5, 1):
            print(f"{i}. {d}  {s:.6f}")


# ---------------------------
# Génère un run
# ---------------------------
def generate_one_run(run_name, method, postings, df, doc_len, doc_ids, N,
                     queries, stopset, stemmer, stem_cache, out_dir):
    ensure_dir(out_dir)
    run_path = os.path.join(out_dir, f"{TEAM}_{run_name}.txt")

    if method == "ltn":
        weighted, _ = compute_ltn_weights(postings, df, N)
    elif method == "ltc":
        weighted, _ = compute_ltc_weights(postings, df, N)
    else:
        weighted = None

    count = 0
    with open(run_path, "w", encoding="utf-8") as f:
        for qid, qtext in queries.items():
            q_terms = preprocess_tokens(tokenizer(qtext), stopset, stemmer, stem_cache)

            if method == "ltn":
                scores = score_query_ltn(weighted, q_terms)
            elif method == "ltc":
                scores = score_query_ltc(weighted, q_terms)
            else:
                scores, _ = score_query_bm25(postings, df, doc_len, N, q_terms)

            ranked = top_k_with_padding(scores, doc_ids, TOP_K)

            for rank, (docid, s) in enumerate(ranked, 1):
                f.write(f"{qid} Q0 {docid} {rank} {s:.5f} {TEAM} /article[1]\n")
                count += 1

    return run_path, count, len(queries) * TOP_K


# ---------------------------
# Main: génération des 12 runs
# ---------------------------
def main():
    start = time.time()
    print("=== Génération des 49 runs ===")

    docs = load_collection(DATAFILE)
    print(f"Documents chargés : {len(docs)}")

    stop_full = load_stopwords(STOPFILE)

    stop_options = [("nostop", set()), ("stop671", stop_full)]
    stem_options = [
        ("nostem", None),
        ("porter", PorterStemmer()),
        ("snowball", SnowballStemmer("english")),
        ("lancaster", LancasterStemmer()),
    ]
    methods = ["ltn", "ltc", "bm25"]

    ensure_dir(OUTPUT_DIR)
    run_paths = []

    run_id = 0

    for stop_name, stopset in stop_options:
        for stem_name, stemmer in stem_options:

            stem_cache = {}
            print(f"\n--- Construction index (stop={stop_name}, stem={stem_name}) ---")

            t0 = time.time()
            postings, df, doc_len, doc_ids, stem_cache, stats = build_index(docs, stopset, stemmer)
            N = len(doc_ids)
            print(f"Index construit en {time.time() - t0:.2f}s — {len(df):,} termes")

            # Affichage TP4 (uniquement aux coordonnées demandées)
            if stop_name == "nostop" and stem_name == "nostem":
                print("\n=== LNT stats (sans stopword, sans stemmer) ===")
                for k, v in stats.items():
                    print(f"{k}: {v}")

            if stop_name == "stop671" and stem_name == "porter":
                print("\n=== LNT stats (avec stopword, avec porter) ===")
                for k, v in stats.items():
                    print(f"{k}: {v}")

            # Runs
            for method in methods:
                if method == "bm25":
                    if stem_name == "nostem" or stem_name == "porter":
                        run_name = (
                            f"{run_id}_test_{method}_article_{stop_name}_{stem_name}"
                            f"_k{BM25_K1}_b{BM25_B}"
                        )
                    else:
                        run_name = (
                            f"{run_id}_{method}_article_{stop_name}_{stem_name}"
                            f"_k{BM25_K1}_b{BM25_B}"
                        )
                else:
                    if stem_name == "nostem" or stem_name == "porter":
                        run_name = (
                            f"{run_id}_test_{method}_article_{stop_name}_{stem_name}"
                        )
                    else:
                        run_name = (
                            f"{run_id}_{method}_article_{stop_name}_{stem_name}"
                        )

                print(f"\n-> Génération run {run_name} ...")

                path, written, expected = generate_one_run(
                    run_name, method, postings, df, doc_len, doc_ids, N,
                    QUERIES, stopset, stemmer, stem_cache, OUTPUT_DIR
                )

                print(f"  Fichier : {path} — {written}/{expected}")

                # Analyse spéciale (UNIQUEMENT NOSTOP / NOSTEM)
                if stop_name == "nostop" and stem_name == "nostem":
                    analyse_single_method(method, postings, df, doc_len, N,
                                          stopset, stemmer, stem_cache)

                run_paths.append(path)
                run_id += 1

    method = "bm25"
    stop_name, stopset = "nostop", set()
    stem_name, stemmer = "nostem", None
    k1_values = [0.2, 1.0, 1.8, 2.6, 3.4]
    b_values = [0.0, 0.25, 0.50, 0.75, 1.0]

    print(f"\n--- Construction index (stop={stop_name}, stem={stem_name}) ---")
    t0 = time.time()
    postings, df, doc_len, doc_ids, stem_cache, stats = build_index(docs, stopset, stemmer)
    N = len(doc_ids)
    print(f"Index construit en {time.time() - t0:.2f}s — {len(df):,} termes")

    for K1 in k1_values:
        for B in b_values:
            run_name = (
                f"{run_id}_{method}_article_{stop_name}_{stem_name}"
                f"_k{K1}_b{B}"
            )

            print(f"-> Génération run tuning {run_name} ...")

            path, written, expected = generate_one_run(
                run_name, method, postings, df, doc_len, doc_ids, N,
                QUERIES, stopset, stemmer, stem_cache, OUTPUT_DIR, K1, B
            )

            print(f"  Fichier : {path} — {written}/{expected}")

            run_paths.append(path)
            run_id += 1

    # ZIP final
    zipname = f"{TEAM}_ALL_RUNS.zip"
    zip_path = os.path.join(OUTPUT_DIR, zipname)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in run_paths:
            zf.write(p, os.path.basename(p))
    print(f"\nZIP généré : {zip_path}")

    end = time.time()
    elapsed = end - start
    print(elapsed)


if __name__ == "__main__":
    main()
