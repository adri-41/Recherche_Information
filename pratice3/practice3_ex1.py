import re
import os
import time

def read_documents(text):
    pattern = re.compile(r"<doc>\s*<docno>\s*([^<\s]+)\s*</docno>(.*?)</doc>",
                         flags=re.IGNORECASE | re.DOTALL)
    docs = []
    for m in pattern.finditer(text):
        doc_id = m.group(1).strip()
        contained = m.group(2)
        docs.append((doc_id, contained))
    return docs


def tokenizer_tokens(text):
    """Tokenisation normalisée : respecte la casse (A–Z et a–z)."""
    return re.findall(r"[A-Za-z]+", text)


def tokenizer_terms(text):
    """Tokenisation des termes : minuscules uniquement."""
    return re.findall(r"[a-z]+", text.lower())


def compute_stats(docs):
    total_tokens = 0
    distinct_tokens = set()
    total_token_chars = 0

    total_terms = 0
    distinct_terms = set()
    total_term_chars = 0

    doc_lengths = []

    for _, content in docs:
        tokens = tokenizer_tokens(content)
        terms = tokenizer_terms(content)

        doc_lengths.append(len(tokens))

        # Statistiques sur les tokens
        total_tokens += len(tokens)
        distinct_tokens.update(tokens)
        total_token_chars += sum(len(t) for t in tokens)

        # Statistiques sur les termes
        total_terms += len(terms)
        distinct_terms.update(terms)
        total_term_chars += sum(len(t) for t in terms)

    avg_doc_length = total_tokens / len(docs) if docs else 0
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

    data_path = os.path.join(os.getcwd(), r"Practice_03_data", "Text_Only_Ascii_Coll_NoSem")
    if not os.path.exists(data_path):
        print(f"Fichier introuvable : {data_path}")
        return

    print("Lecture du fichier...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Extraction des documents...")
    docs = read_documents(text)
    print(f"{len(docs)} documents détectés.")

    print("Calcul des statistiques...")
    stats = compute_stats(docs)

    end = time.time()
    elapsed = end - start

    print("\n===== Résultats de l'indexation =====")
    print(f"Temps total d'indexation : {elapsed:.2f} secondes")
    print(f"Total #tokens : {stats['total_tokens']:,}")
    print(f"Total #distinct tokens : {stats['distinct_tokens']:,}")
    print(f"Moyenne longueur tokens : {stats['avg_token_length']:.2f} caractères")
    print(f"Total #terms : {stats['total_terms']:,}")
    print(f"Total #distinct terms : {stats['distinct_terms']:,}")
    print(f"Longueur moyenne des documents : {stats['avg_doc_length']:.2f} termes")
    print(f"Moyenne longueur des termes : {stats['avg_term_length']:.2f} caractères")
    print("=====================================\n")


if __name__ == "__main__":
    main()