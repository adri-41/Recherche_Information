import re
import os
import time
from nltk.stem import PorterStemmer

def read_documents(text):
    pattern = re.compile(r"<doc>\s*<docno>\s*([^<\s]+)\s*</docno>(.*?)</doc>",
                         flags=re.IGNORECASE | re.DOTALL)
    return [(m.group(1).strip(), m.group(2)) for m in pattern.finditer(text)]


def tokenizer_terms(text):
    """Retourne les termes (normalisés en minuscules)."""
    return re.findall(r"[a-z]+", text.lower())


def tokenizer_tokens(text):
    """Retourne les tokens (respecte la casse)."""
    return re.findall(r"[A-Za-z]+", text)


def compute_stats(docs, stopwords):
    ps = PorterStemmer()

    total_tokens = 0
    distinct_tokens = set()
    total_token_chars = 0

    total_terms = 0
    distinct_terms = set()
    total_term_chars = 0
    doc_lengths = []

    # Cache pour éviter de restemmer les mêmes mots
    stem_cache = {}

    for _, content in docs:
        # Séparation tokens (pour stats) et termes (pour vocabulaire)
        tokens = tokenizer_tokens(content)
        terms = tokenizer_terms(content)

        total_tokens += len(tokens)
        distinct_tokens.update(tokens)
        total_token_chars += sum(len(t) for t in tokens)

        # Traitement stopwords + stemming optimisé
        processed = []
        for tok in tokens:
            if tok in stopwords:
                continue
            # Utiliser le cache pour accélérer
            if tok not in stem_cache:
                stem_cache[tok] = ps.stem(tok)
            processed.append(stem_cache[tok])

        total_terms += len(processed)
        distinct_terms.update(processed)
        total_term_chars += sum(len(t) for t in processed)
        doc_lengths.append(len(processed))

    avg_doc_length = total_terms / len(docs) if docs else 0
    avg_token_length = total_token_chars / total_tokens if total_tokens else 0
    avg_term_length = total_term_chars / total_terms if total_terms else 0

    return {
        "total_tokens": total_tokens,
        "distinct_tokens": len(distinct_tokens),
        "avg_token_length": avg_token_length,
        "total_terms": total_terms,
        "distinct_terms": len(distinct_terms),
        "avg_doc_length": avg_doc_length,
        "avg_term_length": avg_term_length
    }


def main():
    start = time.time()

    data_path = os.path.join("Practice_03_data", "Text_Only_Ascii_Coll_NoSem")
    stopword_path = os.path.join("Practice_03_data", "stop-words-english4.txt")

    if not os.path.exists(data_path):
        print(f"Fichier introuvable : {data_path}")
        return
    if not os.path.exists(stopword_path):
        print(f"Fichier introuvable : {stopword_path}")
        return

    print("Lecture du fichier de collection...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Lecture du fichier de stop-words...")
    with open(stopword_path, "r", encoding="utf-8") as f:
        stopwords = set(line.strip().lower() for line in f if line.strip())

    docs = read_documents(text)
    print(f"{len(docs)} documents détectés.")

    print("Calcul des statistiques avec suppression des stop-words et stemming (optimisé)...")
    stats = compute_stats(docs, stopwords)

    end = time.time()
    elapsed = end - start

    print("\n===== Résultats de l'indexation (stopwords + stemmer) =====")
    print(f"Temps total d'indexation : {elapsed:.2f} secondes")
    print(f"Total #tokens : {stats['total_tokens']:,}")
    print(f"Total #distinct tokens : {stats['distinct_tokens']:,}")
    print(f"Moyenne longueur tokens : {stats['avg_token_length']:.2f} caractères")
    print(f"Total #terms : {stats['total_terms']:,}")
    print(f"Total #distinct terms : {stats['distinct_terms']:,}")
    print(f"Longueur moyenne des documents : {stats['avg_doc_length']:.2f} termes")
    print(f"Moyenne longueur des termes : {stats['avg_term_length']:.2f} caractères")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
