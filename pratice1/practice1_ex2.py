import re
import sys


def read_documents(text):
    pattern = re.compile(r"<doc>\s*<docno>\s*([^<\s]+)\s*</docno>(.*?)</doc>",
                         flags=re.IGNORECASE | re.DOTALL)
    docs = []

    # Iterate over all found <doc>…</doc> blocks
    for m in pattern.finditer(text):
        doc_id = m.group(1).strip()
        contained = m.group(2)
        docs.append((doc_id, contained))
    return docs


def tokeniser(text):
    t = text.lower()
    t = t.replace("’", " ").replace("‘", " ").replace("`", " ")
    return re.findall(r"[a-z0-9]+", t)


# Builds the inverted index
def build_index(docs):
    index = {}
    for doc_id, contained in docs:
        for word in tokeniser(contained):
            if word not in index:
                index[word] = {}
            if doc_id not in index[word]:
                index[word][doc_id] = 0
            index[word][doc_id] += 1
    return index


def key_sort(doc_id):
    m = re.match(r"[A-Za-z]+(\d+)$", doc_id)
    if m:
        return (int(m.group(1)), doc_id)
    return (10 ** 9, doc_id)


# The terms are sorted alphabetically.
def print_index(index):
    for term in sorted(index.keys()):
        postings = index[term]
        df = len(postings)
        print(f"{df}=df({term})")
        for doc_id in sorted(postings.keys(), key=key_sort):
            print(f"{postings[doc_id]} {doc_id}")


def main():
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    docs = read_documents(text)
    index = build_index(docs)
    print_index(index)


if __name__ == "__main__":
    main()

# To launch the program:  python practice1_ex2.py collection.txt
