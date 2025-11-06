import re
import gzip
import time
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


def tokeniser(text):
    t = text.lower()
    t = t.replace("’", " ").replace("‘", " ").replace("`", " ")
    return re.findall(r"[a-z]+", t)


def build_index_and_stats(docs):
    index = {}
    total_terms = 0
    total_chars = 0

    for doc_id, contained in docs:
        tokens = tokeniser(contained)
        total_terms += len(tokens)
        total_chars += sum(len(word) for word in tokens)

        for word in tokens:
            if word not in index:
                index[word] = {}
            index[word][doc_id] = index[word].get(doc_id, 0) + 1

    avg_doc_length = total_terms / len(docs) if docs else 0
    avg_term_length = total_chars / total_terms if total_terms else 0
    vocab_size = len(index)

    stats = {
        "avg_doc_length": avg_doc_length,
        "avg_term_length": avg_term_length,
        "vocab_size": vocab_size
    }

    return index, stats


def read_file(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def main():
    print("Recherche des fichiers de collection dans le dossier Practice_02_data...")

    data_dir = os.path.join(os.getcwd(), "Practice_02_data")
    if not os.path.isdir(data_dir):
        print(f"Dossier introuvable : {data_dir}")
        return

    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".gz")]
    if not files:
        print("Aucun fichier .gz trouvé dans", data_dir)
        return

    print(f"{len(files)} fichiers trouvés : {', '.join(os.path.basename(f) for f in files)}")

    print_index_flag = "--no-print" not in os.sys.argv

    # Listes pour les statistiques et le temps
    sizes = []
    times = []
    avg_doc_lengths = []
    avg_term_lengths = []
    vocab_sizes = []

    for path in files:
        print(f"\nIndexation de {os.path.basename(path)}...")
        start = time.time()

        text = read_file(path)
        docs = read_documents(text)
        index, stats = build_index_and_stats(docs)

        elapsed = time.time() - start
        file_size_kb = os.path.getsize(path) / 1024

        sizes.append(file_size_kb)
        times.append(elapsed)
        avg_doc_lengths.append(stats["avg_doc_length"])
        avg_term_lengths.append(stats["avg_term_length"])
        vocab_sizes.append(stats["vocab_size"])

        print(f"Terminé en {elapsed:.2f} s ({file_size_kb:.0f} KB)")
        print(f"Statistiques : avg_doc_length={stats['avg_doc_length']:.2f}, "
              f"avg_term_length={stats['avg_term_length']:.2f}, vocab_size={stats['vocab_size']}")

        if print_index_flag and file_size_kb < 200:
            print("\n--- Index du fichier ---")
            for term in sorted(index.keys()):
                postings = index[term]
                df = len(postings)
                print(f"{df}=df({term})")
                for doc_id in sorted(postings.keys()):
                    print(f"{postings[doc_id]} {doc_id}")

    # --- Graphique temps / taille ---
    plt.figure(figsize=(6, 4))
    plt.plot(sizes, times, marker="o")
    plt.xlabel("Taille de la collection (KB)")
    plt.ylabel("Temps d'indexation (s)")
    plt.title("Temps d'indexation en fonction de la taille de la collection")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
