import os
import sys
import zipfile
from nltk.stem import PorterStemmer

from practice3_ex3 import (
    compute_ltn_weights,
    score_query_ltn,
    load_collection,
    preprocess_terms,
    tokenizer,
    load_stopwords,
    build_tf_df,
)
from practice3_ex4 import compute_ltc_weights, score_ltc_docs_lnn_query
from practice3_ex5 import score_query_bm25

# DummyStemmer pour les cas sans stemming
class DummyStemmer:
    def stem(self, t):
        return t


# PARAMÈTRES GÉNÉRAUX
queries = {
    "2009011": "olive oil health benefit",
    "2009036": "notting hill film actors",
    "2009067": "probabilistic models in information retrieval",
    "2009073": "web link network analysis",
    "2009074": "web ranking scoring algorithm",
    "2009078": "supervised machine learning algorithm",
    "2009085": "operating system mutual exclusion"
}

RUNS_OUTPUT_DIR = "generated_runs"
os.makedirs(RUNS_OUTPUT_DIR, exist_ok=True)

TEAM = "AdrienSoleneWilliam"
DATA_DIR = os.path.join(os.path.dirname(__file__), "Practice_03_data")
DATAFILE = os.path.join(DATA_DIR, "Text_Only_Ascii_Coll_NoSem")
STOPFILE = os.path.join(DATA_DIR, "stop-words-english4.txt")


# FONCTION DE GÉNÉRATION
def generate_run(run_id, method, stopwords, stemmer, stem_cache, postings, df, doc_len, N, stop, stem, doc_ids):
    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  → Génération run {run_id}  ({method}, stop={stop}, stem={stem})")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Pré-calcul des poids si nécessaire
    try:
        if method == "ltn":
            weighted_postings, _ = compute_ltn_weights(postings, df, N)
        elif method == "ltc":
            weighted_postings, _ = compute_ltc_weights(postings, df, N)
        else:
            weighted_postings = None  # BM25 n’en a pas besoin
    except Exception as e:
        print(f"[CRITICAL] Erreur lors du calcul des poids ({method}) : {e}")
        return None

    run_filename = f"{TEAM}_{run_id}_{method}_article_{stop}_{stem}.txt"
    run_path = os.path.join(RUNS_OUTPUT_DIR, run_filename)

    total_lines_written = 0
    all_doc_ids = list(doc_ids)

    try:
        with open(run_path, "w", encoding="utf-8") as f:

            for idx, (qid, qtext) in enumerate(queries.items(), start=1):
                print(f"    → Query {qid} ({idx}/{len(queries)})")

                q_tokens = preprocess_terms(tokenizer(qtext), stopwords, stemmer, stem_cache)

                # SCORING
                try:
                    if method == "ltn":
                        scores = score_query_ltn(weighted_postings, q_tokens)

                    elif method == "ltc":
                        scores = score_ltc_docs_lnn_query(weighted_postings, q_tokens)

                    elif method == "bm25":
                        scores, _ = score_query_bm25(
                            postings,
                            df,
                            doc_len,
                            N,
                            q_tokens,
                            1.2,
                            0.75,
                        )

                except Exception as e:
                    print(f"      [ERREUR] Scoring query {qid} : {e}")
                    continue

                print(f"      [DEBUG] {method} – {qid} : {len(scores)} docs scorés")

                # TOP 1500 (ou moins si scores vide)
                top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1500]

                # 🔴 COMPLÉTION : si moins de 1500 docs, on rajoute des docs avec score 0
                if len(top_docs) < 1500:
                    print(f"      [WARN] Seulement {len(top_docs)} docs pour {qid}, on complète à 1500.")
                    used_docs = {doc_id for doc_id, _ in top_docs}
                    missing = 1500 - len(top_docs)

                    for doc_id in all_doc_ids:
                        if doc_id not in used_docs:
                            top_docs.append((doc_id, 0.0))
                            used_docs.add(doc_id)
                            if len(top_docs) == 1500:
                                break

                for rank, (doc_id, score) in enumerate(top_docs, start=1):
                    f.write(f"{qid} Q0 {doc_id} {rank} {score:.5f} {TEAM} /article[1]\n")
                    total_lines_written += 1

    except Exception as e:
        print(f"[CRITICAL] ERREUR PENDANT ÉCRITURE : {e}")
        return run_path

    expected = len(queries) * 1500
    print(f"    ✔ Run terminé : {run_filename}")
    print(f"    → Lignes écrites : {total_lines_written}/{expected}")

    if total_lines_written != expected:
        print(f"    ⚠ AVERTISSEMENT : fichier incomplet ! ({total_lines_written}/{expected})")

    return run_path


# CHARGEMENT DE LA COLLECTION
collection = load_collection(DATAFILE)
docs_text_only = [(doc_id, content) for doc_id, content in collection]
print(f"{len(docs_text_only)} documents chargés.")


# COMBINAISONS STOP / STEM
stop_options = ["nostop", "stop671"]
stem_options = ["nostem", "porter"]
methods = ["ltn", "ltc", "bm25"]

stopwords_dict = {
    "nostop": set(),
    "stop671": load_stopwords(STOPFILE)
}


# GÉNÉRATION DE TOUS LES RUNS
run_paths = []
run_id_count = 1

for stop in stop_options:
    print(f"\n=== STOP OPTION : {stop} ===")

    for stem in stem_options:
        print(f"\n--- STEM OPTION : {stem} ---")

        stopwords = stopwords_dict[stop]

        stem_cache = {}
        if stem == "porter":
            stemmer = PorterStemmer()
        else:
            stemmer = DummyStemmer()

        print("  → Construction postings/df...")
        postings, df, doc_ids, _, _ = build_tf_df(docs_text_only, stopwords)
        N = len(doc_ids)

        # OPTIMISATION DU CALCUL DOC_LEN
        print("  → Calcul doc_len (optimisé)...")
        doc_len = {doc: 0 for doc in doc_ids}
        for term, plist in postings.items():
            for doc, tf in plist.items():
                doc_len[doc] += tf

        # Génération des méthodes
        for method in methods:
            print(f"      Méthode {method} en cours…")

            run_path = generate_run(
                run_id_count,
                method,
                stopwords,
                stemmer,
                stem_cache,
                postings,
                df,
                doc_len,
                N,
                stop,
                stem,
                doc_ids
            )
            run_paths.append(run_path)
            run_id_count += 1


# ZIP FINAL
zip_name = f"{TEAM}_ALL_RUNS.zip"
zip_path = os.path.join(RUNS_OUTPUT_DIR, zip_name)

print("\nCréation du ZIP final...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for path in run_paths:
        if path:
            zipf.write(path, os.path.basename(path))

print(f"ZIP généré : {zip_path}")