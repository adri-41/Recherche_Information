import re
import gzip
import os
import matplotlib.pyplot as plt

def read_documents(text):
    pattern = re.compile(r"<doc>\s*<docno>\s*([^<\s]+)\s*</docno>(.*?)</doc>",
                         flags=re.IGNORECASE | re.DOTALL)
    docs = []
    for m in pattern.finditer(text):
        doc_id = m.group(1).strip()
        contained = m.group(2)
        docs.append((doc_id, contained))
    return docs

def read_file(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

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
    toks = tokeniser(text)
    return [t for t in toks if t not in stopset]

def build_stats(docs, tokenizer):
    total_terms = 0
    total_chars = 0
    vocab = set()
    for _, contained in docs:
        tokens = tokenizer(contained)
        total_terms += len(tokens)
        total_chars += sum(len(w) for w in tokens)
        vocab.update(tokens)
    avg_doc_length = total_terms / len(docs) if docs else 0.0
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

    files = sorted([os.path.join(data_dir, f)
                    for f in os.listdir(data_dir) if f.endswith(".gz")])

    if not files:
        print("Aucun fichier .gz trouvé dans", data_dir)
        return

    file9 = next((p for p in files if os.path.basename(p).startswith("09-")), None)
    if file9:
        text9 = read_file(file9)
        docs9 = read_documents(text9)

        avg_doc_b, avg_term_b, vocab_b, tokens_b = build_stats(docs9, tokeniser)
        avg_doc_s, avg_term_s, vocab_s, tokens_s = build_stats(docs9, lambda t: tokeniser_stopwords(t, stopset))

        print("[Exo 3.1] Fichier 9 (sans / avec stopwords)")
        print(f"  SANS stopwords -> avg_doc_len={avg_doc_b:.2f}, avg_term_len={avg_term_b:.2f}, "
              f"vocab={vocab_b}, tokens={tokens_b}")
        print(f"  AVEC stopwords -> avg_doc_len={avg_doc_s:.2f}, avg_term_len={avg_term_s:.2f}, "
              f"vocab={vocab_s}, tokens={tokens_s}")
        print()

    # Évolution par fichier avec stopwords
    total_terms_list = []
    avg_doc_lengths = []
    avg_term_lengths = []
    vocab_sizes = []

    for path in files:
        print(f"Traitement (stopwords) de {os.path.basename(path)}...")
        text = read_file(path)
        docs = read_documents(text)
        avg_doc, avg_term, vocab_size, total_terms = build_stats(docs, lambda t: tokeniser_stopwords(t, stopset))

        total_terms_list.append(total_terms)               
        avg_doc_lengths.append(round(avg_doc, 2))
        avg_term_lengths.append(round(avg_term, 2))
        vocab_sizes.append(vocab_size)

        print(f"Mots (après stopwords): {total_terms} | Longueur moy doc: {round(avg_doc, 2)} | "
              f"Longueur moy terme: {round(avg_term, 2)} | Vocabulaire: {vocab_size}")

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(total_terms_list, avg_doc_lengths, marker="o")
    plt.xlabel("Nombre total de mots dans le fichier (après stopwords)")
    plt.ylabel("Longueur moyenne des documents")
    plt.title("1. Longueur moyenne des documents (stopwords)")
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='both')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 2)
    plt.plot(total_terms_list, avg_term_lengths, marker="s")
    plt.xlabel("Nombre total de mots dans le fichier (après stopwords)")
    plt.ylabel("Longueur moyenne des termes (caractères)")
    plt.title("2. Longueur moyenne des termes (stopwords)")
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='both')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    plt.plot(total_terms_list, vocab_sizes, marker="^")
    plt.xlabel("Nombre total de mots dans le fichier (après stopwords)")
    plt.ylabel("Taille du vocabulaire (mots distincts)")
    plt.title("3. Taille du vocabulaire (stopwords)")
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='both')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
