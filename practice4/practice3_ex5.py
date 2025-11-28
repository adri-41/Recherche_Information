import re
import os
import time
import math
import argparse
from collections import defaultdict
from nltk.stem import PorterStemmer

# --- CONSTANTES ---
# Paramètres spécifiques à l'Exercice 5 (BM25)
K1 = 1.2
B = 0.75

DOC_PATTERN = re.compile(r"<doc>\s*<docno>\s*([^<\s]+)\s*</docno>(.*?)</doc>",
                         flags=re.IGNORECASE | re.DOTALL)


# --- FONCTIONS GÉNÉRIQUES (Base Ex 2, 3, 4) ---

def read_documents(text):
    """Générateur pour lire les documents à partir du texte de la collection."""
    for m in DOC_PATTERN.finditer(text):
        doc_id = m.group(1).strip()
        content = m.group(2)
        yield doc_id, content


def tokenizer(text):
    """Tokenisation simple: mots alphabétiques en minuscules."""
    return re.findall(r"[a-z]+", text.lower())


def process_tokens(tokens, stopwords, ps):
    """Suppression des stop-words et application du Stemming."""
    terms = []
    for tok in tokens:
        if tok not in stopwords:
            terms.append(ps.stem(tok))
    return terms


def build_tf_df_and_lengths(docs_iter, stopwords, ps):
    """
    Construit l'index inversé (postings), le df, et enregistre les longueurs
    de document (en termes après traitement).
    """
    postings = defaultdict(lambda: defaultdict(int))
    df = defaultdict(int)
    doc_lengths = {}
    N = 0

    for docno, content in docs_iter:
        N += 1
        tokens = tokenizer(content)
        terms = process_tokens(tokens, stopwords, ps)

        doc_lengths[docno] = len(terms)  # Longueur après traitement

        seen = set()
        for t in terms:
            postings[t][docno] += 1
            if t not in seen:
                df[t] += 1
                seen.add(t)

    return postings, df, doc_lengths, N


# --- FONCTION BM25 ---

def score_query_bm25(postings, df, doc_lengths, N, q_tokens, k1, b):
    """
    Calcule le RSV pour tous les documents en utilisant le modèle BM25.

    RSV(d, q) = SUM [ IDF(t) * ( (tf_td * (k1 + 1)) / (tf_td + k1 * (1-b + b*dl_d/avdl)) ) ]
    """

    # Calcul de la longueur moyenne des documents (avdl)
    if not doc_lengths:
        return {}  # Aucun document, pas de score

    avdl = sum(doc_lengths.values()) / len(doc_lengths)

    # Calcul de l'IDF (logarithme naturel est standard pour BM25)
    # IDF(t) = ln( (N - df_t + 0.5) / (df_t + 0.5) )
    idf = {t: math.log((N - df_t + 0.5) / (df_t + 0.5))
           for t, df_t in df.items() if df_t > 0 and df_t < N}  # df_t < N pour éviter log(0)

    scores = defaultdict(float)

    # La requête est déjà tokénisée et stemmée/filtrée.
    # On n'a pas besoin des TF de la requête (comme dans ltn/ltc) en BM25.

    # Parcourir les termes de la requête
    for t in set(q_tokens):
        if t not in postings:
            continue

        idf_t = idf.get(t, 0.0)
        if idf_t <= 0:  # Si l'IDF est non pertinent (terme trop fréquent ou inconnu)
            continue

        # Parcourir les documents qui contiennent le terme t
        for docno, tf_td in postings[t].items():
            dl_d = doc_lengths.get(docno, avdl)  # Longueur du document

            # Facteur de normalisation de longueur: k1 * ( (1-b) + b * (dl_d / avdl) )
            normalization_factor = k1 * ((1 - b) + b * (dl_d / avdl))

            # Calcul du TF ajusté (TF_adj): tf_td * (k1 + 1) / (tf_td + normalization_factor)
            tf_adj = (tf_td * (k1 + 1)) / (tf_td + normalization_factor)

            # Ajout de la contribution du terme au score du document (RSV)
            scores[docno] += idf_t * tf_adj

    return scores, avdl  # On retourne avdl pour le debug si besoin


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Ranking with BM25 (k1=1.2, b=0.75).")
    parser.add_argument("--query", type=str,
                        default="web ranking scoring algorithm", help="La requête de recherche.")
    parser.add_argument("--docno", type=str,
                        default="23724", help="Le docno cible pour l'analyse.")
    parser.add_argument("--report", type=str,
                        default="practice3_report.txt", help="Nom du fichier de rapport.")
    args = parser.parse_args()

    # Chemins basés sur la structure fournie dans les exercices précédents
    base_dir = os.path.join(os.getcwd(), "Practice_03_data")
    data_path = os.path.join(base_dir, "Text_Only_Ascii_Coll_NoSem")
    stopword_path = os.path.join(base_dir, "stop-words-english4.txt")

    if not os.path.exists(data_path) or not os.path.exists(stopword_path):
        print(
            f"Erreur: Assurez-vous que les fichiers de données ({os.path.basename(data_path)} et {os.path.basename(stopword_path)}) existent.")
        return

    # 1. Préparation de l'index
    print("Lecture et indexation des documents (avec stop-words et stemming)...")
    start = time.time()

    ps = PorterStemmer()
    with open(stopword_path, "r", encoding="utf-8") as f:
        stopwords = set(line.strip().lower() for line in f if line.strip())

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    docs_iter = read_documents(text)

    # posts: postings list, df: document frequency, doc_len: document lengths, N: collection size
    postings, df, doc_lengths, N = build_tf_df_and_lengths(docs_iter, stopwords, ps)

    index_time = time.time() - start
    print(f"Indexation terminée en {index_time:.3f} secondes. N={N} documents.")

    # 2. Préparation de la requête
    q_tokens_raw = tokenizer(args.query)
    q_tokens = process_tokens(q_tokens_raw, stopwords, ps)

    # 3. Classement (Scoring)
    print("Calcul des scores BM25...")
    weighting_start = time.time()

    scores, avdl = score_query_bm25(postings, df, doc_lengths, N, q_tokens, K1, B)

    weighting_time = time.time() - weighting_start

    # 4. Extraction des résultats demandés
    target = args.docno
    term = "ranking"  # Le terme 'ranking' doit être stemmé pour l'index
    stemmed_term = ps.stem(term)

    # Calcul du Poids du terme (TF_adj) dans le document cible (utilisé dans BM25)
    w_ranking_target = 0.0

    if stemmed_term in postings and target in postings[stemmed_term]:
        tf_td = postings[stemmed_term][target]
        dl_d = doc_lengths.get(target, avdl)

        # Partie de normalisation
        normalization_factor = K1 * ((1 - B) + B * (dl_d / avdl))
        tf_adj = (tf_td * (K1 + 1)) / (tf_td + normalization_factor)

        # Poids final du terme = IDF * TF_adj (ce qui est la contribution du terme)
        # On peut choisir de reporter juste le TF_adj ou la contribution totale.
        # Ici, je retourne le TF_adj qui est le "poids" du terme dans le document.
        w_ranking_target = tf_adj

    rsv_target = scores.get(target, 0.0)

    # Tri des documents pour le Top-10
    top10 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10]

    # 5. Affichage et rapport
    print("\n===== Résultats du Classement (BM25) =====")
    print(f"Collection size (N): {N}")
    print(f"Avg Document Length (avdl): {avdl:.2f}")
    print(f"Paramètres BM25: k1={K1}, b={B}")
    print(f"Total weighting time : {weighting_time:.3f} sec")
    print(f'Query: "{args.query}"  (Termes traités: {q_tokens})')
    print(f'Poids du terme "{term}" stemmé ("{stemmed_term}") dans doc={target} [TF_adj] = {w_ranking_target:.6f}')
    print(f'RSV(doc={target}) = {rsv_target:.6f}')
    print("\nTop-10 documents:")
    for rank, (d, s) in enumerate(top10, start=1):
        print(f"{rank:2d}. doc={d}  RSV={s:.6f}")

    # Écriture dans le rapport
    try:
        with open(args.report, "a", encoding="utf-8") as rep:
            rep.write("\n\n####################################################\n")
            rep.write("### EXERCICE 5: RANKED RETRIEVAL (BM25 weighting) ###\n")
            rep.write("####################################################\n")
            rep.write(f"Collection size (N): {N}\n")
            rep.write(f"Avg Document Length (avdl): {avdl:.2f}\n")
            rep.write(f"Paramètres BM25: k1={K1}, b={B}\n")
            rep.write(f"Total weighting time : {weighting_time:.3f} sec\n")
            rep.write(f'Query: "{args.query}"  (Termes traités: {q_tokens})\n')
            rep.write(
                f'Poids du terme "{term}" stemmé ("{stemmed_term}") dans doc={target} [TF_adj] = {w_ranking_target:.6f}\n')
            rep.write(f'RSV(doc={target}) = {rsv_target:.6f}\n')
            rep.write("\nTop-10 documents:\n")
            for rank, (d, s) in enumerate(top10, start=1):
                rep.write(f"{rank:2d}. doc={d}  RSV={s:.6f}\n")
    except Exception as e:
        print(f"\nErreur lors de l'écriture du rapport : {e}")


if __name__ == "__main__":
    main()
