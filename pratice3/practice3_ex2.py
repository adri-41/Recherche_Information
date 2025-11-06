import re
import os
import time
from nltk.stem import PorterStemmer

def read_documents(text):
    pattern = re.compile(r"<doc>\s*<docno>\s*([^<\s]+)\s*</docno>(.*?)</doc>",
                         flags=re.IGNORECASE | re.DOTALL)
    docs = []
    for m in pattern.finditer(text):
        doc_id = m.group(1).strip()
        contained = m.group(2)
        docs.append((doc_id, contained))
    return docs

def tokenizer(text):
    """
    Simple tokenization: only alphabetic words, lowercase.
    """
    return re.findall(r"[a-z]+", text.lower())

def compute_stats(docs, stopwords):
    ps = PorterStemmer()

    total_tokens = 0
    distinct_tokens = set()
    total_token_chars = 0

    total_terms = 0
    distinct_terms = set()
    total_term_chars = 0
    doc_lengths = []

    for _, content in docs:
        tokens = tokenizer(content)

        total_tokens += len(tokens)
        distinct_tokens.update(tokens)
        total_token_chars += sum(len(t) for t in tokens)

        processed = []
        for tok in tokens:
            if tok not in stopwords:
                stemmed = ps.stem(tok)
                processed.append(stemmed)

        total_terms += len(processed)
        distinct_terms.update(processed)
        total_term_chars += sum(len(t) for t in processed)
        doc_lengths.append(len(processed))

    avg_doc_length = total_terms / len(docs) if docs else 0
    avg_token_length = total_token_chars / len(distinct_tokens) if distinct_tokens else 0
    avg_term_length = total_term_chars / len(distinct_terms) if distinct_terms else 0

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

    data_path = r"Practice_03_data\Text_Only_Ascii_Coll_NoSem"
    stopword_path = r"Practice_03_data\stop-words-english4.txt"

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

    print("Calcul des statistiques avec suppression des stop-words et stemming...")
    stats = compute_stats(docs, stopwords)

    end = time.time()
    elapsed = end - start

    print("\n===== Résultats de l'indexation (avec stopwords + stemmer) =====")
    print(f"Temps total d'indexation : {elapsed:.2f} secondes")
    print(f"Total #tokens : {stats['total_tokens']:,}")
    print(f"Total #distinct tokens : {stats['distinct_tokens']:,}")
    print(f"Moyenne longueur tokens : {stats['avg_token_length']:.2f} caractères")
    print(f"Total #terms : {stats['total_terms']:,}")
    print(f"Total #distinct terms : {stats['distinct_terms']:,}")
    print(f"Longueur moyenne des documents : {stats['avg_doc_length']:.2f} termes")
    print(f"Moyenne longueur des termes : {stats['avg_term_length']:.2f} caractères")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
