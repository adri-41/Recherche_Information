import re
import gzip
import os
import time
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


def tokeniser(text):
    t = text.lower()
    t = t.replace("’", " ").replace("‘", " ").replace("`", " ")
    return re.findall(r"[a-z]+", t)


def build_stats(docs):
    total_terms = 0
    total_chars = 0
    vocab = set()

    for doc_id, contained in docs:
        tokens = tokeniser(contained)
        total_terms += len(tokens)
        total_chars += sum(len(w) for w in tokens)
        vocab.update(tokens)

    avg_doc_length = total_terms / len(docs) if docs else 0
    avg_term_length = total_chars / total_terms if total_terms else 0
    vocab_size = len(vocab)

    return avg_doc_length, avg_term_length, vocab_size, total_terms


def read_file(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def main():
    start_time = time.time()  # --- Début du chronomètre ---

    data_dir = os.path.join(os.getcwd(), "Practice_02_data")
    if not os.path.isdir(data_dir):
        print(f"Dossier introuvable : {data_dir}")
        return

    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".gz")])

    total_terms_list = []
    avg_doc_lengths = []
    avg_term_lengths = []
    vocab_sizes = []

    for path in files:
        print(f"Traitement de {os.path.basename(path)}...")
        text = read_file(path)
        docs = read_documents(text)
        avg_doc, avg_term, vocab_size, total_terms = build_stats(docs)

        total_terms_list.append(total_terms)
        avg_doc_lengths.append(round(avg_doc, 2))
        avg_term_lengths.append(round(avg_term, 2))
        vocab_sizes.append(vocab_size)

        print(f"Mots: {total_terms} | Longueur moyenne doc: {round(avg_doc, 2)} | "
              f"Longueur moyenne terme: {round(avg_term, 2)} | Vocabulaire: {vocab_size}")

    # --- Tracer 3 sous-graphes ---
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(total_terms_list, avg_doc_lengths, marker="o")
    plt.xlabel("Nombre total de mots dans le fichier")
    plt.ylabel("Longueur moyenne des documents")
    plt.title("1. Longueur moyenne des documents")
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='both')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 2)
    plt.plot(total_terms_list, avg_term_lengths, marker="s", color="orange")
    plt.xlabel("Nombre total de mots dans le fichier")
    plt.ylabel("Longueur moyenne des termes (caractères)")
    plt.title("2. Longueur moyenne des termes")
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='both')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    plt.plot(total_terms_list, vocab_sizes, marker="^", color="green")
    plt.xlabel("Nombre total de mots dans le fichier")
    plt.ylabel("Taille du vocabulaire (mots distincts)")
    plt.title("3. Taille du vocabulaire")
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='both')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    end_time = time.time()  # --- Fin du chronomètre ---
    elapsed = end_time - start_time
    print(f"\nTemps total d'exécution : {round(elapsed, 2)} secondes")


if __name__ == "__main__":
    main()
