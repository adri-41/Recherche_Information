
import re
import os
import time
import math
import argparse
from collections import defaultdict, Counter

DOC_PATTERN = re.compile(r"<doc>\s*<docno>\s*([^<\s]+)\s*</docno>(.*?)</doc>",
                         flags=re.IGNORECASE | re.DOTALL)

def read_documents(text):
    for m in DOC_PATTERN.finditer(text):
        doc_id = m.group(1).strip()
        content = m.group(2)
        yield doc_id, content

def tokenizer(text):
    return re.findall(r"[a-z]+", text.lower())

def build_tf_df(docs_iter):
    postings = defaultdict(lambda: defaultdict(int))
    df = defaultdict(int)
    doc_ids = []
    for docno, content in docs_iter:
        doc_ids.append(docno)
        terms = tokenizer(content)
        seen = set()
        for t in terms:
            postings[t][docno] += 1
            if t not in seen:
                df[t] += 1
                seen.add(t)
    return postings, df, doc_ids

def compute_ltc_doc_weights(postings, df, N):
    idf = {t: math.log10(N / df_t) for t, df_t in df.items() if df_t > 0}
    weights = defaultdict(dict)
    squared_len = defaultdict(float)
    for t, doc_tf in postings.items():
        idf_t = idf.get(t, 0.0)
        if idf_t <= 0:
            continue
        for d, tf_td in doc_tf.items():
            if tf_td <= 0:
                continue
            w = (1.0 + math.log10(tf_td)) * idf_t
            weights[t][d] = w
            squared_len[d] += w * w
    norms = {d: math.sqrt(sq) if sq > 0 else 1.0 for d, sq in squared_len.items()}
    for t, docs in list(weights.items()):
        for d, w in list(docs.items()):
            docs[d] = w / norms[d]
    return weights, idf, norms

def compute_ltc_query_weights(idf, query_tokens):
    q_tf = Counter(query_tokens)
    wq = {}
    for t, tf_tq in q_tf.items():
        if tf_tq <= 0:
            continue
        idf_t = idf.get(t, 0.0)
        if idf_t <= 0:
            continue
        wq[t] = (1.0 + math.log10(tf_tq)) * idf_t
    norm_q = math.sqrt(sum(v*v for v in wq.values())) or 1.0
    for t in list(wq.keys()):
        wq[t] /= norm_q
    return wq

def score_query_ltc(weighted_postings_ltc, wq):
    scores = defaultdict(float)
    for t, w_tq in wq.items():
        postings_t = weighted_postings_ltc.get(t, {})
        for d, w_td in postings_t.items():
            scores[d] += w_td * w_tq
    return scores

def load_collection(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return list(read_documents(text))

def main():
    parser = argparse.ArgumentParser(description="Exercise 4: SMART ltc cosine retrieval")
    parser.add_argument("--data", default=os.path.join(os.getcwd(), r"Practice_03_data", "Text_Only_Ascii_Coll_NoSem"),
                        help="Chemin vers le fichier de collection (concatenated docs).")
    parser.add_argument("--docno", default="23724", help="Docno pour l'inspection ciblée (par défaut 23724).")
    parser.add_argument("--query", default="web ranking scoring algorithm", help="Requête à scorer.")
    parser.add_argument("--report", default="practice3_report.txt", help="Fichier où ajouter un extrait de résultats.")
    args = parser.parse_args()

    t0 = time.time()
    docs = load_collection(args.data)
    N = len(docs)
    postings, df, doc_ids = build_tf_df(docs)
    weighted_postings_ltc, idf, norms = compute_ltc_doc_weights(postings, df, N)
    weighting_time = time.time() - t0

    q_tokens = tokenizer(args.query)
    wq = compute_ltc_query_weights(idf, q_tokens)
    scores = score_query_ltc(weighted_postings_ltc, wq)

    target = args.docno
    term = "ranking"
    w_ranking_target = weighted_postings_ltc.get(term, {}).get(target, 0.0)
    rsv_target = scores.get(target, 0.0)

    top10 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10]

    print(f"Collection size (N): {N}")
    print(f"Total weighting time : {weighting_time:.3f} sec")
    print(f'Query: "{args.query}"  (tokens: {q_tokens})')
    print(f'Weight(term=\"ranking\", doc={target}) [ltc] = {w_ranking_target:.6f}')
    print(f'RSV(doc={target}) [cosine] = {rsv_target:.6f}')
    print("\nTop-10 documents:")
    for rank, (d, s) in enumerate(top10, start=1):
        print(f"{rank:2d}. doc={d}  RSV={s:.6f}")

    try:
        with open(args.report, "a", encoding="utf-8") as rep:
            rep.write(f"Collection size (N): {N}\n")
            rep.write(f"Total weighting time : {weighting_time:.3f} sec\n")
            rep.write(f'Query: "{args.query}"  (tokens: {q_tokens})\n')
            rep.write(f'Weight(term="ranking", doc={target}) [ltc] = {w_ranking_target:.6f}\n')
            rep.write(f'RSV(doc={target}) [cosine] = {rsv_target:.6f}\n')
            rep.write("Top-10 documents:\n")
            for rank, (d, s) in enumerate(top10, start=1):
                rep.write(f"{rank:2d}. doc={d}  RSV={s:.6f}\n")
            rep.write("\n")
    except Exception as e:
        print(f"[WARN] Impossible d'écrire dans le rapport: {e}")

if __name__ == "__main__":
    main()
