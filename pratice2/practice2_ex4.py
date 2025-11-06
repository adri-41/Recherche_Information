import gzip
import os
import re

import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer


def iter_documents(path):
    """Version optimisée : lit tout le fichier et extrait les documents d'un coup."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    pattern = re.compile(r"<doc>\s*<docno>\s*([^<\s]+)\s*</docno>(.*?)</doc>", flags=re.IGNORECASE | re.DOTALL)
    for doc_id, contained in pattern.findall(text):
        yield doc_id.strip(), contained



def tokeniser(text):
    t = text.lower().replace("’", " ").replace("‘", " ").replace("`", " ")
    return re.findall(r"[a-z]+", t)


def load_stopwords(path):
    stop = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip().lower()
            if s and not s.startswith("#"):
                stop.add(s)
    return stop


def tokeniser_stopwords(text, stopset):
    return [t for t in tokeniser(text) if t not in stopset]


def tokeniser_stopwords_stem(text, stopset, stemmer):
    return [stemmer.stem(t) for t in tokeniser_stopwords(text, stopset)]


def build_stats_iter(docs_iter, tokenizer):
    """Calcule les stats sans stocker tous les tokens en mémoire"""
    total_terms = 0
    total_chars = 0
    vocab = set()
    n_docs = 0

    for _, contained in docs_iter:
        tokens = tokenizer(contained)
        n_docs += 1
        total_terms += len(tokens)
        total_chars += sum(len(t) for t in tokens)
        vocab.update(tokens)

    avg_doc_length = total_terms / n_docs if n_docs else 0.0
    avg_term_length = total_chars / total_terms if total_terms else 0.0
    vocab_size = len(vocab)
    return avg_doc_length, avg_term_length, vocab_size, total_terms


def main():
    data_dir = os.path.join(os.getcwd(), "Practice_02_data")
    stop_path = os.path.join(data_dir, "stop-words-english4.txt")

    if not os.path.isdir(data_dir):
        print(f"Dossier introuvable : {data_dir}")
        return
    if not os.path.exists(stop_path):
        print(f"Fichier stop-words introuvable : {stop_path}")
        return

    stopset = load_stopwords(stop_path)
    stemmer = PorterStemmer()

    files = sorted([os.path.join(data_dir, f)
                    for f in os.listdir(data_dir) if f.endswith(".gz")])

    if not files:
        print("Aucun fichier .gz trouvé dans", data_dir)
        return

    file9 = next((p for p in files if os.path.basename(p).startswith("09-")), None)
    if file9:
        # docs_iter = iter_documents(file9)
        docs = list(iter_documents(file9))  # lire le fichier UNE SEULE FOIS

        avg_doc_stop, avg_term_stop, vocab_stop, tokens_stop = build_stats_iter(
            docs, lambda t: tokeniser_stopwords(t, stopset)
        )
        avg_doc_stem, avg_term_stem, vocab_stem, tokens_stem = build_stats_iter(
            docs, lambda t: tokeniser_stopwords_stem(t, stopset, stemmer)
        )

        print("[Exo 4.1] Fichier 9")
        print(f"  Stopwords -> avg_doc_len={avg_doc_stop:.2f}, avg_term_len={avg_term_stop:.2f}, "
              f"vocab={vocab_stop}, tokens={tokens_stop}")
        print(f"  Stopwords + Stemmer -> avg_doc_len={avg_doc_stem:.2f}, avg_term_len={avg_term_stem:.2f}, "
              f"vocab={vocab_stem}, tokens={tokens_stem}")
        print()

    total_terms_list = []
    avg_doc_lengths = []
    avg_term_lengths = []
    vocab_sizes = []

    total_terms_stem = []
    avg_doc_stem_list = []
    avg_term_stem_list = []
    vocab_stem_list = []

    for path in files:
        print(f"Traitement de {os.path.basename(path)}...")

        # Stopwords
        avg_doc, avg_term, vocab_size, total_terms = build_stats_iter(
            iter_documents(path), lambda t: tokeniser_stopwords(t, stopset)
        )
        # Stopwords + Stemmer
        avg_doc_s, avg_term_s, vocab_size_s, total_terms_s = build_stats_iter(
            iter_documents(path), lambda t: tokeniser_stopwords_stem(t, stopset, stemmer)
        )

        total_terms_list.append(total_terms)
        avg_doc_lengths.append(round(avg_doc, 2))
        avg_term_lengths.append(round(avg_term, 2))
        vocab_sizes.append(vocab_size)

        total_terms_stem.append(total_terms_s)
        avg_doc_stem_list.append(round(avg_doc_s, 2))
        avg_term_stem_list.append(round(avg_term_s, 2))
        vocab_stem_list.append(vocab_size_s)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(total_terms_list, avg_doc_lengths, "o-", label="Stopwords")
    plt.plot(total_terms_stem, avg_doc_stem_list, "s--", label="Stopwords + Stemmer")
    plt.xlabel("Nombre total de mots")
    plt.ylabel("Longueur moyenne des documents")
    plt.title("Longueur moyenne des documents")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(total_terms_list, avg_term_lengths, "o-", label="Stopwords")
    plt.plot(total_terms_stem, avg_term_stem_list, "s--", label="Stopwords + Stemmer")
    plt.xlabel("Nombre total de mots")
    plt.ylabel("Longueur moyenne des termes")
    plt.title("Longueur moyenne des termes")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(total_terms_list, vocab_sizes, "o-", label="Stopwords")
    plt.plot(total_terms_stem, vocab_stem_list, "s--", label="Stopwords + Stemmer")
    plt.xlabel("Nombre total de mots")
    plt.ylabel("Taille du vocabulaire")
    plt.title("Taille du vocabulaire")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
