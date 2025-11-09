import os
import re
import time
import math
import argparse
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer

DOC_PATTERN = re.compile(r"<doc>\s*<docno>\s*([^<\s]+)\s*</docno>(.*?)</doc>",
                         flags=re.IGNORECASE | re.DOTALL)
TOKEN_PATTERN = re.compile(r"[a-z]+")

def read_documents(text):
    for m in DOC_PATTERN.finditer(text):
        yield m.group(1).strip(), m.group(2)

def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())

def load_stopwords(path):
    if not path:
        return set()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

def preprocess_tokens(tokens, stopwords, stemmer, stem_cache):
    out = []
    for tok in tokens:
        if tok in stopwords:
            continue
        if tok not in stem_cache:
            stem_cache[tok] = stemmer.stem(tok)
        out.append(stem_cache[tok])
    return out

def l_weight(tf):
    return 0.0 if tf <= 0 else 1.0 + math.log10(tf)

def idf_weight(N, df):
    return 0.0 if df <= 0 else math.log10(N / df)

def build_tf_df(text, stopwords):
    ps = PorterStemmer()
    stem_cache = {}
    df = defaultdict(int)  
    doc_tfs = {}              
    N = 0
    for doc_id, content in read_documents(text):
        N += 1
        terms = preprocess_tokens(tokenize(content), stopwords, ps, stem_cache)
        c = Counter(terms)
        doc_tfs[doc_id] = c
        for t in c.keys():
            df[t] += 1
    return doc_tfs, df, N, ps, stem_cache

def compute_ltc_weights(doc_tfs, df, N):
   
    idf = {t: idf_weight(N, df_t) for t, df_t in df.items()}

    raw = {}
    for d, tf_counts in doc_tfs.items():
        w = {}
        for t, tf in tf_counts.items():
            w[t] = l_weight(tf) * idf.get(t, 0.0)
        raw[d] = w

    ltc = {}
    for d, w in raw.items():
        norm_sq = sum(val * val for val in w.values())
        if norm_sq <= 0:
            ltc[d] = {}
            continue
        norm = math.sqrt(norm_sq)
        ltc[d] = {t: (val / norm) for t, val in w.items()}
    return ltc, idf

def score_ltc_docs_lnn_query(doc_weights, q_tokens):

    q_tf = Counter(q_tokens)
    q_w = {t: l_weight(tf) for t, tf in q_tf.items() if tf > 0}
    scores = defaultdict(float)

    for t, wqt in q_w.items():
        for d, w_td in ((d, w.get(t)) for d, w in doc_weights.items() if t in w):
            scores[d] += w_td * wqt
    return scores

def main():
    ap = argparse.ArgumentParser(description="Exercise 4: SMART ltc ranked retrieval (stopwords + Porter)")
    ap.add_argument("---data", default=os.path.join(os.path.dirname(__file__), "Practice_03_data", "Text_Only_Ascii_Coll_NoSem"),
                    help="Chemin vers le fichier de collection.")
    ap.add_argument("--stop", default=os.path.join(os.path.dirname(__file__), "Practice_03_data", "stop-words-english4.txt"),
                    help="Chemin vers le fichier de stop-words.")
    ap.add_argument("--docno", default="23724", help="Docno pour inspection ciblée (par défaut 23724).")
    ap.add_argument("--query", default="web ranking scoring algorithm", help="Requête à scorer.")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERREUR] Fichier introuvable : {args.data}")
        return
    if not os.path.exists(args.stop):
        print(f"[ERREUR] Fichier introuvable : {args.stop}")
        return

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    stopwords = load_stopwords(args.stop)
    doc_tfs, df, N, ps, stem_cache = build_tf_df(text, stopwords)

    t0 = time.time()

    doc_ltc, idf = compute_ltc_weights(doc_tfs, df, N)
    q_tokens = preprocess_tokens(tokenize(args.query), stopwords, ps, stem_cache)
    scores = score_ltc_docs_lnn_query(doc_ltc, q_tokens)

    weighting_time = time.time() - t0

    target = args.docno

    term_raw = "ranking"
    term_token = term_raw.lower()
    term_stem = None if term_token in stopwords else ps.stem(term_token)

    w_ranking_target = 0.0
    if term_stem and target in doc_ltc:
        w_ranking_target = doc_ltc[target].get(term_stem, 0.0)

    rsv_target = scores.get(target, 0.0)

    top10 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10]

    print("===== Exercise 4: SMART ltc =====")
    print(f"Collection size (N): {N}")
    print(f"Total weighting time: {weighting_time:.2f} sec")
    print(f'Query: "{args.query}"  (tokens after preprocess: {q_tokens})')
    print(f'Weight(term="ranking", doc={target}) = {w_ranking_target:.6f}')
    print(f'RSV(doc={target}) = {rsv_target:.6f}')
    print("\nTop-10 documents:")
    for rank, (d, s) in enumerate(top10, start=1):
        print(f"{rank:2d}. doc={d}  RSV={s:.6f}")

if __name__ == "__main__":
    main()
