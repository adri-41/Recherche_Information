import os
import re
import time
import math
import zipfile
from bs4 import BeautifulSoup
from lxml import etree
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer


# =======================
# Config
# =======================
DATA_DIR = os.path.join(os.path.dirname(__file__), "Practice_05_data", "XML-Coll-withSem")
STOPFILE = os.path.join(os.path.dirname(__file__), "Practice_03_data", "stop-words-english4.txt")
DATAFILE = os.path.join(DATA_DIR, "XML-Coll-withSem")

OUTPUT_DIR = "generated_runs"
TEAM = "AdrienSoleneWilliam"

QUERIES = {
    "2009011": "olive oil health benefit",
    "2009036": "notting hill film actors",
    "2009067": "probabilistic models in information retrieval",
    "2009073": "web link network analysis",
    "2009074": "web ranking scoring algorithm",
    "2009078": "supervised machine learning algorithm",
    "2009085": "operating system mutual exclusion"
}

TOP_K = 1500

BM25_K1 = 1.2
BM25_B = 0.75

# Deux regex distinctes : tokens (pour stats, casse conservée) et terms (pour indexation, minuscules)
TOKEN_RE = re.compile(r"[A-Za-z]+")  # pour tokens (stats)
TERM_RE = re.compile(r"[a-z]+")      # pour terms (indexation)
TAG_RE = re.compile(r"<[^>]+>")


# =======================
# Utilitaires
# =======================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_stopwords(path):
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {w.strip().lower() for w in f if w.strip()}


DOC_RE = re.compile(
    r"<doc>\s*<docno>\s*([^<]+)\s*</docno>(.*?)</doc>",
    flags=re.IGNORECASE | re.DOTALL
)


def load_collection(path):
    """Charge la collection entière en liste (docid, contenu brut)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [(m.group(1).strip(), m.group(2)) for m in DOC_RE.finditer(text)]


def load_collection_xml(root_dir):
    """
    Parcourt tous les fichiers .xml dans root_dir
    et renvoie une liste (docid, contenu_texte_sans_balises).
    """
    docs = []
    for fname in os.listdir(root_dir):
        if not fname.lower().endswith(".xml"):
            continue
        path = os.path.join(root_dir, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        text_no_tags = TAG_RE.sub(" ", raw)
        docid = os.path.splitext(fname)[0]  
        docs.append((docid, text_no_tags))
    return docs


# =======================
# Tokenisation / Preprocessing
# =======================
def tokenize_tokens(text):
    """Tokenisation pour stats : respecte la casse (A-Za-z)."""
    return TOKEN_RE.findall(text)


def tokenize_terms(text):
    """Tokenisation pour termes (indexation/requêtes) : minuscules uniquement."""
    return TERM_RE.findall(text.lower())


def preprocess(tokens, stopset, stemmer, cache):
    """Supprime stopwords et applique stemming via stemmer (utilise cache)."""
    out = []
    for t in tokens:
        if t in stopset:
            continue
        if stemmer is None:
            out.append(t)
        else:
            if t not in cache:
                cache[t] = stemmer.stem(t)
            out.append(cache[t])
    return out


# =======================
# Construction de l’index (corrigée pour tokens vs terms)
# =======================
def build_index(docs, stopset, stemmer):
    """
    Construit postings, df, doc_len, doc_ids, stem_cache et stats.
    Stats distinctes pour tokens (casse) et terms (minuscules/stemmés).
    """
    postings = defaultdict(lambda: defaultdict(int))
    df = defaultdict(int)
    doc_len = {}
    doc_ids = []
    stem_cache = {}

    # stats tokens
    total_tokens = 0
    total_token_chars = 0
    distinct_tokens = set()

    # stats terms
    total_terms = 0
    total_term_chars = 0
    distinct_terms = set()

    for docid, content in docs:
        doc_ids.append(docid)

        # tokens pour stats (respect de la casse)
        tokens = tokenize_tokens(content)
        total_tokens += len(tokens)
        total_token_chars += sum(len(t) for t in tokens)
        distinct_tokens.update(tokens)

        # terms pour index (minuscules) puis preprocess (stop/stem)
        raw_terms = tokenize_terms(content)  # already lowercased
        terms = preprocess(raw_terms, stopset, stemmer, stem_cache)

        doc_len[docid] = len(terms)
        total_terms += len(terms)
        total_term_chars += sum(len(t) for t in terms)
        distinct_terms.update(terms)

        # construire postings & df
        tf = Counter(terms)
        for term, freq in tf.items():
            postings[term][docid] = freq
            df[term] += 1

    stats = {
        "total_tokens": total_tokens,
        "distinct_tokens": len(distinct_tokens),
        "avg_token_len": (total_token_chars / total_tokens) if total_tokens else 0,
        "total_terms": total_terms,
        "distinct_terms": len(distinct_terms),
        "avg_doc_len": (total_terms / len(doc_ids)) if doc_ids else 0,
        "avg_term_len": (total_term_chars / total_terms) if total_terms else 0,
    }

    return postings, df, doc_len, doc_ids, stem_cache, stats


# =======================
# LTN / LTC weights
# =======================
def compute_ltn_weights(postings, df, N):
    weights = defaultdict(dict)
    for t, plist in postings.items():
        df_t = df[t]
        if df_t <= 0:
            continue
        idf_t = math.log10(N / df_t)
        for d, tf in plist.items():
            weights[t][d] = (1 + math.log10(tf)) * idf_t
    return weights


def compute_ltc_weights(postings, df, N):
    raw = defaultdict(dict)
    norm_sq = Counter()
    for t, plist in postings.items():
        df_t = df[t]
        if df_t <= 0:
            continue
        idf_t = math.log10(N / df_t)
        for d, tf in plist.items():
            w = (1 + math.log10(tf)) * idf_t
            raw[t][d] = w
            norm_sq[d] += w * w
    weights = defaultdict(dict)
    for t, plist in raw.items():
        for d, w in plist.items():
            norm = math.sqrt(norm_sq[d])
            weights[t][d] = w / norm if norm > 0 else 0.0
    return weights


# =======================
# Scoring LTN / LTC / BM25
# =======================
def score_query_ltn(weights, query_terms):
    q_tf = Counter(query_terms)
    q_w = {t: 1 + math.log10(tf) for t, tf in q_tf.items()}
    scores = defaultdict(float)
    for t, wq in q_w.items():
        for d, wtd in weights.get(t, {}).items():
            scores[d] += wtd * wq
    return scores


def score_query_ltc(weights, query_terms):
    q_tf = Counter(query_terms)
    q_w = {t: 1 + math.log10(tf) for t, tf in q_tf.items()}
    scores = defaultdict(float)
    for t, wq in q_w.items():
        for d, wtd in weights.get(t, {}).items():
            scores[d] += wtd * wq
    return scores


def score_query_bm25(postings, df, doc_len, N, query_terms, k1=1.2, b=0.75):
    if not doc_len:
        return {}, 0.0
    avdl = sum(doc_len.values()) / len(doc_len)
    scores = defaultdict(float)
    for t in set(query_terms):
        if t not in postings:
            continue
        df_t = df.get(t, 0)
        idf_t = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1e-12)
        for d, tf in postings[t].items():
            dl = doc_len[d]
            denom = tf + k1 * ((1 - b) + b * (dl / avdl))
            tf_adj = (tf * (k1 + 1)) / denom if denom > 0 else 0.0
            scores[d] += idf_t * tf_adj
    return scores, avdl


# =======================
# Helper top-k avec padding
# =======================
def top_k_with_padding(scores, doc_ids, k=TOP_K):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) >= k:
        return ranked[:k]
    used = set(d for d, _ in ranked)
    pad = [(d, 0.0) for d in doc_ids if d not in used][:k - len(ranked)]
    return ranked + pad


# =======================
# Génération d'un run (assure bien la transmission k1/b)
# =======================
def generate_one_run(run_name, method, postings, df, doc_len, doc_ids, N,
                     queries, stopset, stemmer, stem_cache, out_dir,
                     k1=BM25_K1, b=BM25_B):
    ensure_dir(out_dir)
    run_path = os.path.join(out_dir, f"{TEAM}_{run_name}.txt")
    with open(run_path, "w", encoding="utf-8") as f:
        if method == "ltn":
            weights = compute_ltn_weights(postings, df, N)
        elif method == "ltc":
            weights = compute_ltc_weights(postings, df, N)
        else:
            weights = None

        count = 0
        for qid, qtext in queries.items():
            q_raw_tokens = tokenize_terms(qtext)          # utiliser tokenisation terms pour requêtes
            q_terms = preprocess(q_raw_tokens, stopset, stemmer, stem_cache)

            if method == "ltn":
                scores = score_query_ltn(weights, q_terms)
            elif method == "ltc":
                scores = score_query_ltc(weights, q_terms)
            else:
                scores, _ = score_query_bm25(postings, df, doc_len, N, q_terms, k1=k1, b=b)

            ranked = top_k_with_padding(scores, doc_ids, TOP_K)
            for rank, (docid, s) in enumerate(ranked, 1):
                f.write(f"{qid} Q0 {docid} {rank} {s:.5f} {TEAM} /article[1]\n")
                count += 1

    expected = len(queries) * TOP_K
    return run_path, count, expected

def get_element_path(node):
    path_parts = []
    while node is not None and node.name is not None:
        # Compte la position parmi les frères du même tag
        if node.parent:
            siblings = [s for s in node.parent.find_all(node.name, recursive=False)]
            index = siblings.index(node) + 1
        else:
            index = 1
        path_parts.insert(0, f"{node.name}[{index}]")
        node = node.parent
    return "/" + "/".join(path_parts)

def extract_elements_xml(filepath, tags=("bdy","sec","p")):
    """
    Retourne une liste (element_id, text, path_in_xml) pour chaque élément XML choisi.
    element_id = filename + "_" + tag + "_" + compteur
    """
    elements = []
    fname = os.path.splitext(os.path.basename(filepath))[0]
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            xml_content = f.read()
        soup = BeautifulSoup(xml_content, "xml")
    except Exception as e:
        print(f"Erreur parsing {filepath}: {e}")
        return elements

    article = soup.find("article")
    if not article:
        return elements

    counter = {}
    for tag in tags:
        counter[tag] = 1
        for node in article.find_all(tag):
            text = node.get_text(" ", strip=True)
            if text:
                element_id = f"{fname}_{tag}_{counter[tag]}"

                # Calculer le chemin complet depuis <article>
                path_parts = []
                current = node
                while current != article:
                    siblings = [s for s in current.parent.find_all(current.name, recursive=False)]
                    index = siblings.index(current) + 1
                    path_parts.insert(0, f"{current.name}[{index}]")
                    current = current.parent
                path_in_xml = "/article[1]/" + "/".join(path_parts)

                elements.append((element_id, text, path_in_xml))
                counter[tag] += 1
    return elements

def load_collection_elements(root_dir, tags=("bdy","sec","p")):
    """
    Parcourt tous les fichiers .xml et retourne une liste d'éléments
    [(element_id, text, path_in_xml), ...]
    """
    elements = []
    for fname in os.listdir(root_dir):
        if not fname.lower().endswith(".xml"):
            continue
        path = os.path.join(root_dir, fname)
        file_elements = extract_elements_xml(path, tags=tags)
        if not file_elements:
            print(f"Aucun élément trouvé dans {path}")
        elements.extend(file_elements)
    return elements

def generate_elements_run(run_name, postings, df, doc_len, doc_ids, N,
                          queries, stopset, stemmer, stem_cache, out_dir,
                          element_paths=None):
    """
    Génère un fichier de run pour les éléments XML avec SMART LTN.
    element_paths : dictionnaire {element_id: path_in_xml}
    """
    ensure_dir(out_dir)
    run_path = os.path.join(out_dir, f"{TEAM}_{run_name}.txt")

    # Calcul des poids LTN
    weights = compute_ltn_weights(postings, df, N)

    # Normalisation LTN
    norm_sq = {}
    for t, plist in weights.items():
        for d, w in plist.items():
            norm_sq[d] = norm_sq.get(d, 0.0) + w*w
    norm = {d: math.sqrt(v) for d, v in norm_sq.items()}

    weights_ltn = {}
    for t, plist in weights.items():
        for d, w in plist.items():
            weights_ltn.setdefault(t, {})[d] = (w / norm[d]) if norm.get(d,0) > 0 else 0.0

    with open(run_path, "w", encoding="utf-8") as f:
        for qid, qtext in queries.items():
            q_raw = tokenize_terms(qtext)
            q_terms = preprocess(q_raw, stopset, stemmer, stem_cache)
            scores = score_query_ltn(weights_ltn, q_terms)
            ranked = top_k_with_padding(scores, doc_ids, TOP_K)
            for rank, (docid, score) in enumerate(ranked, 1):
                # Récupération du chemin XML depuis element_paths
                path_in_xml = element_paths.get(docid, "/article[1]") if element_paths else "/article[1]"
                f.write(f"{qid} Q0 {docid} {rank} {score:.5f} {TEAM} {path_in_xml}\n")

    print(f"Run éléments généré: {run_path}")
    return run_path

def main_elements_run():
    print("\n=== Exercise 3: Indexing XML elements (bdy, sec, p) ===")
    docs_elements = load_collection_elements(DATA_DIR, tags=("bdy","sec","p"))
    print(f"Nombre total d'éléments extraits: {len(docs_elements)}")

    # Transformer docs_elements pour build_index
    docs_elements_index = [(eid, text) for eid, text, path in docs_elements]
    element_paths = {eid: path for eid, _, path in docs_elements}  # dictionnaire docid → path_in_xml

    # Construire l'index (nostop / nostem)
    stopset = set()
    stemmer = None
    stem_cache = {}

    # Transformer docs_elements pour build_index
    docs_elements_index = [(eid, text) for eid, text, path in docs_elements]

    postings, df, doc_len, doc_ids, stem_cache, stats = build_index(docs_elements_index, stopset, stemmer)
    N = len(doc_ids)
    print(f"Index éléments construit: {len(df)} termes, {len(doc_ids)} éléments")

    # Générer run
    run_name = "12_testXML_ltn_element-bdy-sec-p_nostop_nostem"
    generate_elements_run(
        run_name, postings, df, doc_len, doc_ids, N,
        QUERIES, stopset, stemmer, stem_cache, OUTPUT_DIR,
        element_paths=element_paths
    )

# =======================
# Main : runs exercice 4 + tuning BM25 (grille 5x5)
# =======================
def main():
    start = time.time()
    print("=== Génération des runs (exercice 2 : XML docs, 12 runs) ===")

    docs = load_collection_xml(DATA_DIR)
    print(f"Documents chargés : {len(docs)}")
    
    stop_full = load_stopwords(STOPFILE)

    stop_options = [
        ("nostop", set()),
        ("stop671", stop_full),
    ]
    stem_options = [
        ("nostem", None),
        ("porter", PorterStemmer()),
    ]
    methods = ["ltn", "ltc", "bm25"]

    ensure_dir(OUTPUT_DIR)
    run_paths = []
    run_id = 0

    # ==========================
    # Exercice 2 : runs "test2"
    # ==========================
    for stop_name, stopset in stop_options:
        for stem_name, stemmer in stem_options:
            print(f"\n--- Construction index (stop={stop_name}, stem={stem_name}) ---")
            stem_cache = {}
            t0 = time.time()
            postings, df, doc_len, doc_ids, stem_cache, stats = build_index(docs, stopset, stemmer)
            N = len(doc_ids)
            print(f"Index construit en {time.time() - t0:.2f}s — {len(df):,} termes")

            if stop_name == "nostop" and stem_name == "nostem":
                print("\n=== Stats clés (nostop / nostem) ===")
                for k, v in stats.items():
                    print(f"{k}: {v}")

            for method in methods:
                if method == "bm25":

                    run_name = f"{run_id}_test2_{method}_article_{stop_name}_{stem_name}_k{BM25_K1}_b{BM25_B}"
                    path, written, expected = generate_one_run(
                        run_name, method,
                        postings, df, doc_len, doc_ids, N,
                        QUERIES, stopset, stemmer, stem_cache, OUTPUT_DIR,
                        k1=BM25_K1, b=BM25_B
                    )
                else:
                    run_name = f"{run_id}_test2_{method}_article_{stop_name}_{stem_name}"
                    path, written, expected = generate_one_run(
                        run_name, method,
                        postings, df, doc_len, doc_ids, N,
                        QUERIES, stopset, stemmer, stem_cache, OUTPUT_DIR
                    )

                print(f"  Fichier : {path} — {written}/{expected}")
                run_paths.append(path)
                run_id += 1

    print(f"\nTotal de fichiers de runs générés : {len(run_paths)} (devrait être 12)")
    end = time.time()
    print(f"Temps total: {end - start:.2f}s")

    # ==========================
    # Exercice 3 : runs "testXML"
    # ==========================
    main_elements_run()

if __name__ == "__main__":
    main()

