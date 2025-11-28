import re
import os
import time
import math
import argparse
from collections import defaultdict, Counter

try:
    from nltk.stem import PorterStemmer
except Exception:
    PorterStemmer = None

DOC_PATTERN = re.compile(r"<doc>\s*<docno>\s*([^<\s]+)\s*</docno>(.*?)</doc>",
                         flags=re.IGNORECASE | re.DOTALL)

TOKEN_PATTERN = re.compile(r"[a-z]+")


def read_documents(text):
    for m in DOC_PATTERN.finditer(text):
        doc_id = m.group(1).strip()
        content = m.group(2)
        yield doc_id, content


def tokenizer(text):
    return TOKEN_PATTERN.findall(text.lower())


def load_stopwords(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(w.strip().lower() for w in f if w.strip())
    return set()


def get_stemmer():
    if PorterStemmer is None:
        raise RuntimeError("NLTK est requis pour PorterStemmer. Fais: pip install nltk")
    return PorterStemmer()


def preprocess_terms(tokens, stopwords, stemmer, cache):
    out = []
    for t in tokens:
        if t in stopwords:
            continue
        if t not in cache:
            cache[t] = stemmer.stem(t)
        out.append(cache[t])
    return out


def build_tf_df(docs_iter, stopwords):
    stemmer = get_stemmer()
    stem_cache = {}
    postings = defaultdict(lambda: defaultdict(int)) 
    df = defaultdict(int)                             
    doc_ids = []

    for docno, content in docs_iter:
        doc_ids.append(docno)
        terms = preprocess_terms(tokenizer(content), stopwords, stemmer, stem_cache)
        seen = set()
        for t in terms:
            postings[t][docno] += 1
        for t in postings:
            pass 

        for t in set(terms):
            df[t] += 1

    return postings, df, doc_ids, stemmer, stem_cache


def compute_ltn_weights(postings, df, N):
    weighted = {}
    idf = {t: (0.0 if df_t <= 0 else math.log10(N / df_t)) for t, df_t in df.items()}
    for t, doc_tf in postings.items():
        w_for_t = {}
        idf_t = idf.get(t, 0.0)
        if idf_t <= 0:
            weighted[t] = {}
            continue
        for d, tf_td in doc_tf.items():
            if tf_td > 0:
                w_for_t[d] = (1.0 + math.log10(tf_td)) * idf_t
        weighted[t] = w_for_t
    return weighted, idf


def score_query_ltn(weighted_postings, query_tokens):
    q_tf = Counter(query_tokens)
    q_w = {t: (1.0 + math.log10(tf)) for t, tf in q_tf.items() if tf > 0}

    scores = defaultdict(float)
    for t, wqt in q_w.items():
        postings_t = weighted_postings.get(t, {})
        for d, w_td in postings_t.items():
            scores[d] += w_td * wqt
    return scores


def load_collection(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return list(read_documents(text))


def main():
    parser = argparse.ArgumentParser(description="Exercise 3: SMART ltn ranked retrieval (avec stopwords + Porter)")
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "Practice_03_data", "Text_Only_Ascii_Coll_NoSem"),
                        help="Chemin vers le fichier de collection (concaténation de docs).")
    parser.add_argument("--stop", default=os.path.join(os.getcwd(), r"Practice_03_data", "stop-words-english4.txt"),
                        help="Chemin vers la liste de stop-words.")
    parser.add_argument("--docno", default="23724", help="Docno pour l'inspection ciblée (par défaut 23724).")
    parser.add_argument("--query", default="web ranking scoring algorithm", help="Requête à scorer.")
    args = parser.parse_args()

    docs = load_collection(args.data)
    N = len(docs)

    stopwords = load_stopwords(args.stop)
    postings, df, doc_ids, stemmer, stem_cache = build_tf_df(docs, stopwords)

    t0 = time.time()

    weighted_postings, idf = compute_ltn_weights(postings, df, N)
    q_terms = preprocess_terms(tokenizer(args.query), stopwords, stemmer, stem_cache)
    scores = score_query_ltn(weighted_postings, q_terms)

    weighting_time = time.time() - t0

    target = args.docno
    term_raw = "ranking"
    term_token = term_raw.lower()
    term_stem = None if term_token in stopwords else stemmer.stem(term_token)
    w_ranking_target = 0.0
    if term_stem:
        w_ranking_target = weighted_postings.get(term_stem, {}).get(target, 0.0)

    rsv_target = scores.get(target, 0.0)

    top10 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10]

    print(f"Collection size (N): {N}")
    print(f"Total weighting time : {weighting_time:.3f} sec")
    print(f'Query: "{args.query}"  (tokens after preprocess: {q_terms})')
    print(f'Weight(term="ranking", doc={target}) = {w_ranking_target:.6f}')
    print(f'RSV(doc={target}) = {rsv_target:.6f}')
    print("\nTop-10 documents:")
    for rank, (d, s) in enumerate(top10, start=1):
        print(f"{rank:2d}. doc={d}  RSV={s:.6f}")


if __name__ == "__main__":
    main()
