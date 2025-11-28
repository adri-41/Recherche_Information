import re
import time
import math
import os
import sys
from collections import Counter
import practice3_ex3
from practice3_ex3 import compute_ltn_weights, score_query_ltn, load_collection, preprocess_terms, tokenizer, load_stopwords, build_tf_df

# CONFIGURATION
data_path = os.path.join("Practice_03_data", "Text_Only_Ascii_Coll_NoSem")

queries = {
    "2009011": "olive oil health benefit",
    "2009036": "notting hill film actors",
    "2009067": "probabilistic models in information retrieval",
    "2009073": "web link network analysis",
    "2009074": "web ranking scoring algorithm",
    "2009078": "supervised machine learning algorithm",
    "2009085": "operating system mutual exclusion"
}
max_docs_per_query = 1500
# Paramètres pour le nom de fichier
team_name = "AdrienSoleneWilliam"
run_id_number = 2
weighting = "ltn"
granularity = "article"
stop_option = "nostop"
stem_option = "nostem"

run_file = f"{team_name}_{run_id_number}_{weighting}_{granularity}_{stop_option}_{stem_option}.txt"

# LECTURE DE LA COLLECTION
start_time = time.time()
docs = load_collection(data_path)
print(f"{len(docs)} documents détectés.")

# CONSTRUCTION DES POSTINGS + DF
stopwords = set()  # pas de stop-words pour ce run
postings, df, doc_ids, stemmer, stem_cache = build_tf_df(docs, stopwords)
N = len(doc_ids)

# CALCUL DES POIDS LTN
weighted_postings, idf = compute_ltn_weights(postings, df, N)

# CALCUL DES SCORES ET GÉNÉRATION DU RUN
run_lines = []

for qid, query_text in queries.items():
    query_terms = tokenizer(query_text)
    query_terms = preprocess_terms(query_terms, stopwords, stemmer, stem_cache)

    # Scores RSV pour tous les documents
    scores = score_query_ltn(weighted_postings, query_terms)

    # Top N documents (1500 max)
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_docs_per_query]

    # Génération des lignes au format INEX (7 colonnes)
    for rank, (doc_id, score) in enumerate(top_docs, start=1):
        xml_path = "/article[1]"  # unité = article entier
        run_lines.append(f"{qid} Q0 {doc_id} {rank} {score:.4f} {team_name} {xml_path}")

# ÉCRITURE DU FICHIER DE RUN
with open(run_file, "w", encoding="utf-8") as f:
    f.write("\n".join(run_lines))

elapsed_time = time.time() - start_time
print(f"Run INEX généré dans {run_file} en {elapsed_time:.2f} secondes.")
