"""
Core IR system logic: document loading, search, and stop word removal.
"""

import urllib.request
import re
import math
from collections import defaultdict
from document import Document


def load_collection_from_url(url, search_pattern, start_line, end_line, author, origin):
    """
    Download a text file and parse it into Document objects using a regex pattern.

    Parameters:
        url (str): URL of the plain text file
        search_pattern (Pattern): RE pattern; group 1 = title, group 2 = story text
        start_line (int): 1-based line number to start reading from
        end_line (int): 1-based line number to stop reading at (inclusive)
        author (str): Author name assigned to each document
        origin (str): Collection title assigned to each document

    Returns:
        list[Document]: Parsed documents in order of appearance
    """
    with urllib.request.urlopen(url) as response:
        raw_bytes = response.read()

    # Decode, handling BOM if present
    text = raw_bytes.decode("utf-8-sig")
    all_lines = text.splitlines()

    # Slice to the desired line range (convert 1-based to 0-based)
    selected_lines = all_lines[start_line - 1: end_line]
    content = "\n".join(selected_lines)

    documents = []
    for doc_id, match in enumerate(re.finditer(search_pattern, content)):
        title = match.group(1).strip()
        story_text = match.group(2)

        # raw_text: full text with line breaks replaced by spaces
        raw_text = " ".join(story_text.split())

        # terms: basic word tokenization (split on whitespace)
        terms = raw_text.split()

        doc = Document(
            document_id=doc_id,
            title=title,
            raw_text=raw_text,
            terms=terms,
            author=author,
            origin=origin,
        )
        documents.append(doc)

    return documents


def linear_boolean_search(term, collection, stopword_filtered=False, stemmed=False):
    """
    Boolean linear scan: return all documents with a relevance score of 1 if the
    term is found, 0 otherwise.

    Parameters:
        term (str): The search term (case-insensitive)
        collection (list[Document]): Documents to search
        stopword_filtered (bool): If True, search in filtered_terms instead of terms
        stemmed (bool): If True, apply Porter stemmer to search term and document terms

    Returns:
        list[tuple[int, Document]]: (score, document) for every document in order
    """
    query_terms = []
    for t in term.split():
        t_norm = _normalize_term(t)
        if stemmed:
            t_norm = stem_term(t_norm)
        if t_norm:
            query_terms.append(t_norm)
            
    if not query_terms:
        return [(0, doc) for doc in collection]

    results = []
    for doc in collection:
        if stopword_filtered:
            # filtered_terms may be a list (instance attr) or a callable method
            ft = doc.filtered_terms
            candidate_terms = ft() if callable(ft) else ft
        else:
            candidate_terms = doc.terms

        processed_doc_terms = set()
        for t in candidate_terms:
            t_norm = _normalize_term(t)
            if stemmed:
                t_norm = stem_term(t_norm)
            if t_norm:
                processed_doc_terms.add(t_norm)
                
        found = True
        for qt in query_terms:
            if qt not in processed_doc_terms:
                found = False
                break
                
        if found:
            score = 1
        else:
            score = 0
        results.append((score, doc))

    return results


def vector_space_search(query, collection, stopword_filtered=False, stemmed=False):
    """
    Search using the Vector Space Model with inverted lists and tf.idf.
    Query weighting follows Salton/Buckley (1988): (0.5 + 0.5 * tf / max_tf) * idf.
    Document weighting: tf * idf.
    """
    if not collection:
        return []

    # 1. Parse and process query terms
    query_terms_raw = query.split()
    query_terms = []
    for t in query_terms_raw:
        t_norm = _normalize_term(t)
        if stemmed:
            t_norm = stem_term(t_norm)
        if t_norm:
            query_terms.append(t_norm)

    if not query_terms:
        # If query is empty, return all docs with score 0
        return [(0.0, doc) for doc in collection]

    # Query term frequencies
    query_tfs = defaultdict(int)
    for t in query_terms:
        query_tfs[t] += 1
    max_query_tf = max(query_tfs.values())

    # 2. Build inverted index and compute DF & lengths
    # inverted_index[term] = list of (doc_id, tf)
    inverted_index = defaultdict(list)
    doc_lengths = {doc.document_id: 0.0 for doc in collection}
    N = len(collection)

    # First pass: build inverted index (tf)
    for doc in collection:
        if stopword_filtered:
            ft = doc.filtered_terms
            candidate_terms = ft() if callable(ft) else ft
        else:
            candidate_terms = doc.terms

        doc_tfs = defaultdict(int)
        for t in candidate_terms:
            t_norm = _normalize_term(t)
            if stemmed:
                t_norm = stem_term(t_norm)
            if t_norm:
                doc_tfs[t_norm] += 1

        for term, tf in doc_tfs.items():
            inverted_index[term].append((doc.document_id, tf))

    # Calculate idf for each term in the collection
    idf_map = {}
    for term, postings in inverted_index.items():
        df = len(postings)
        idf_map[term] = math.log10(N / df) if df > 0 else 0.0

    # Calculate document lengths (L2 norm)
    for term, postings in inverted_index.items():
        idf = idf_map[term]
        for doc_id, tf in postings:
            weight = tf * idf
            doc_lengths[doc_id] += weight * weight

    for doc_id in doc_lengths:
        doc_lengths[doc_id] = math.sqrt(doc_lengths[doc_id])

    # 3. Calculate query vector and dot products
    doc_scores = defaultdict(float)
    
    for term, q_tf in query_tfs.items():
        if term not in inverted_index:
            continue
        
        # Query weight: (0.5 + 0.5 * (tf / max_tf)) * idf
        idf = idf_map[term]
        q_weight = (0.5 + 0.5 * (q_tf / max_query_tf)) * idf
        
        # Traverse inverted list
        for doc_id, doc_tf in inverted_index[term]:
            doc_weight = doc_tf * idf
            doc_scores[doc_id] += q_weight * doc_weight

    # 4. Assemble final results
    results = []
    for doc in collection:
        score = 0.0
        if doc.document_id in doc_scores:
            dot_product = doc_scores[doc.document_id]
            length = doc_lengths[doc.document_id]
            if length > 0:
                score = dot_product / length
        results.append((score, doc))
        
    return results


def precision_recall(retrieved: set, relevant: set) -> tuple[float, float]:
    """Calculate precision and recall for a search query."""
    if not retrieved:
        precision = 0.0
    else:
        precision = len(retrieved.intersection(relevant)) / len(retrieved)
        
    if not relevant:
        recall = 0.0
    else:
        recall = len(retrieved.intersection(relevant)) / len(relevant)
        
    return float(precision), float(recall)


def _normalize_term(term: str) -> str:
    """Lowercase a term, remove non-alphanumeric punctuation (except internal apostrophes)."""
    # Remove newlines/carriage returns
    cleaned = term.replace('\n', ' ').replace('\r', '').lower()
    # Remove any character that isn't a word character (\w) or an apostrophe
    cleaned = re.sub(r'[^\w\']', '', cleaned)
    # Strip leading/trailing apostrophes (e.g. 'hello' -> hello, but don't -> don't)
    return cleaned.strip("'")


def remove_stop_words(terms, stopwords):
    """
    Filter stopwords from a list of terms (case-insensitive).

    Parameters:
        terms (list[str]): Original term list
        stopwords (set[str]): Stop words to remove (lowercase expected)

    Returns:
        list[str]: Filtered and lowercased terms
    """
    stop_lower = {sw.lower() for sw in stopwords}
    filtered = []
    for term in terms:
        normalized = _normalize_term(term)
        if normalized and normalized not in stop_lower:
            filtered.append(normalized)
    return filtered


def remove_stop_words_by_frequency(terms, collection, low_freq, high_freq):
    """
    Filter stopwords from a term list using Crouch's frequency-based method.

    Terms that appear in too many documents (>= high_freq fraction) or too few
    (<= low_freq fraction) are treated as stop words.

    Parameters:
        terms (list[str]): Terms to filter
        collection (list[Document]): Reference collection for frequency computation
        low_freq (float): Rare threshold (0–1); terms at or below this are removed
        high_freq (float): Common threshold (0–1); terms at or above this are removed

    Returns:
        list[str]: Filtered and lowercased terms
    """
    num_docs = len(collection)
    if num_docs == 0:
        return [_normalize_term(t) for t in terms if _normalize_term(t)]

    # Compute collection-wide term frequency (Crouch's method):
    # freq(t) = total occurrences of t / total terms in collection
    term_counts = {}
    total_terms = 0
    for doc in collection:
        for t in doc.terms:
            normalized = _normalize_term(t)
            if normalized:
                term_counts[normalized] = term_counts.get(normalized, 0) + 1
                total_terms += 1

    if total_terms == 0:
        return [_normalize_term(t) for t in terms if _normalize_term(t)]

    stop_words = set()
    for term, count in term_counts.items():
        freq = count / total_terms
        if freq >= high_freq or freq <= low_freq:
            stop_words.add(term)

    filtered = []
    for term in terms:
        normalized = _normalize_term(term)
        if normalized and normalized not in stop_words:
            filtered.append(normalized)

    return filtered


def is_consonant(word, i):
    vowels = "aeiou"
    ch = word[i]
    if ch in vowels:
        return False
    if ch == 'y':
        if i == 0:
            return True
        else:
            return not is_consonant(word, i - 1)
    return True

def measure(word):
    m = 0
    state = 'C'
    if len(word) > 0:
        if is_consonant(word, 0):
            state = 'C'
        else:
            state = 'V'
    
    for i in range(1, len(word)):
        if is_consonant(word, i):
            if state == 'V':
                m += 1
                state = 'C'
        else:
            state = 'V'
    return m

def contains_vowel(word):
    for i in range(len(word)):
        if not is_consonant(word, i):
            return True
    return False

def ends_in_double_consonant(word):
    if len(word) >= 2:
        if word[-1] == word[-2] and is_consonant(word, len(word)-1):
            return True
    return False

def ends_in_cvc(word):
    if len(word) >= 3:
        if is_consonant(word, len(word)-3) and not is_consonant(word, len(word)-2) and is_consonant(word, len(word)-1):
            if word[-1] not in ['w', 'x', 'y']:
                return True
    return False

def step_1a(word):
    if word.endswith("sses"):
        return word[:-4] + "ss"
    elif word.endswith("ies"):
        return word[:-3] + "i"
    elif word.endswith("ss"):
        return word[:-2] + "ss"
    elif word.endswith("s"):
        return word[:-1]
    return word

def step_1b(word):
    if word.endswith("eed"):
        stem = word[:-3]
        if measure(stem) > 0:
            return stem + "ee"
        return word
    elif word.endswith("ed"):
        stem = word[:-2]
        if contains_vowel(stem):
            return step_1b_cleanup(stem)
        return word
    elif word.endswith("ing"):
        stem = word[:-3]
        if contains_vowel(stem):
            return step_1b_cleanup(stem)
        return word
    return word

def step_1b_cleanup(stem):
    if stem.endswith("at"):
        return stem + "e"
    elif stem.endswith("bl"):
        return stem + "e"
    elif stem.endswith("iz"):
        return stem + "e"
    elif ends_in_double_consonant(stem) and not (stem.endswith("l") or stem.endswith("s") or stem.endswith("z")):
        return stem[:-1]
    elif measure(stem) == 1 and ends_in_cvc(stem):
        return stem + "e"
    return stem

def step_1c(word):
    if word.endswith("y"):
        stem = word[:-1]
        if contains_vowel(stem):
            return stem + "i"
    return word

def apply_rules(word, rules):
    for suffix, replacement, condition in rules:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if condition(stem):
                return stem + replacement
            return word
    return word

step_2_rules = [
    ("ational", "ate", lambda s: measure(s) > 0),
    ("tional", "tion", lambda s: measure(s) > 0),
    ("enci", "ence", lambda s: measure(s) > 0),
    ("anci", "ance", lambda s: measure(s) > 0),
    ("izer", "ize", lambda s: measure(s) > 0),
    ("abli", "able", lambda s: measure(s) > 0),
    ("alli", "al", lambda s: measure(s) > 0),
    ("entli", "ent", lambda s: measure(s) > 0),
    ("eli", "e", lambda s: measure(s) > 0),
    ("ousli", "ous", lambda s: measure(s) > 0),
    ("ization", "ize", lambda s: measure(s) > 0),
    ("ation", "ate", lambda s: measure(s) > 0),
    ("ator", "ate", lambda s: measure(s) > 0),
    ("alism", "al", lambda s: measure(s) > 0),
    ("iveness", "ive", lambda s: measure(s) > 0),
    ("fulness", "ful", lambda s: measure(s) > 0),
    ("ousness", "ous", lambda s: measure(s) > 0),
    ("aliti", "al", lambda s: measure(s) > 0),
    ("iviti", "ive", lambda s: measure(s) > 0),
    ("biliti", "ble", lambda s: measure(s) > 0),
    ("xflurti", "xti", lambda s: measure(s) > 0)
]

step_3_rules = [
    ("icate", "ic", lambda s: measure(s) > 0),
    ("ative", "", lambda s: measure(s) > 0),
    ("alize", "al", lambda s: measure(s) > 0),
    ("iciti", "ic", lambda s: measure(s) > 0),
    ("ical", "ic", lambda s: measure(s) > 0),
    ("ful", "", lambda s: measure(s) > 0),
    ("ness", "", lambda s: measure(s) > 0)
]

step_4_rules = [
    ("al", "", lambda s: measure(s) > 1),
    ("ance", "", lambda s: measure(s) > 1),
    ("ence", "", lambda s: measure(s) > 1),
    ("er", "", lambda s: measure(s) > 1),
    ("ic", "", lambda s: measure(s) > 1),
    ("able", "", lambda s: measure(s) > 1),
    ("ible", "", lambda s: measure(s) > 1),
    ("ant", "", lambda s: measure(s) > 1),
    ("ement", "", lambda s: measure(s) > 1),
    ("ment", "", lambda s: measure(s) > 1),
    ("ent", "", lambda s: measure(s) > 1),
    ("ion", "", lambda s: measure(s) > 1 and (s.endswith("s") or s.endswith("t"))),
    ("ou", "", lambda s: measure(s) > 1),
    ("ism", "", lambda s: measure(s) > 1),
    ("ate", "", lambda s: measure(s) > 1),
    ("iti", "", lambda s: measure(s) > 1),
    ("ous", "", lambda s: measure(s) > 1),
    ("ive", "", lambda s: measure(s) > 1),
    ("ize", "", lambda s: measure(s) > 1)
]

def step_5a(word):
    if word.endswith("e"):
        stem = word[:-1]
        m = measure(stem)
        if m > 1:
            return stem
        if m == 1 and not ends_in_cvc(stem):
            return stem
    return word

def step_5b(word):
    m = measure(word)
    if m > 1 and ends_in_double_consonant(word) and word.endswith("l"):
        return word[:-1]
    return word

def stem_term(term):
    # I read the porter.txt file and found your note.
    word = term.lower()
    if len(word) <= 2:
        return word
    
    word = step_1a(word)
    word = step_1b(word)
    word = step_1c(word)
    word = apply_rules(word, step_2_rules)
    word = apply_rules(word, step_3_rules)
    word = apply_rules(word, step_4_rules)
    word = step_5a(word)
    word = step_5b(word)
    return word
