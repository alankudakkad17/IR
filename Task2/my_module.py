"""
Core IR system logic: document loading, search, and stop word removal.
"""

import urllib.request
import re
import string
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


def linear_boolean_search(term, collection, stopword_filtered=False):
    """
    Boolean linear scan: return all documents with a relevance score of 1 if the
    term is found, 0 otherwise.

    Parameters:
        term (str): The search term (case-insensitive)
        collection (list[Document]): Documents to search
        stopword_filtered (bool): If True, search in filtered_terms instead of terms

    Returns:
        list[tuple[int, Document]]: (score, document) for every document in order
    """
    term_lower = term.lower()
    results = []

    for doc in collection:
        if stopword_filtered:
            # filtered_terms may be a list (instance attr) or a callable method
            ft = doc.filtered_terms
            candidate_terms = ft() if callable(ft) else ft
        else:
            candidate_terms = doc.terms

        found = any(t.lower() == term_lower for t in candidate_terms)
        score = 1 if found else 0
        results.append((score, doc))

    return results


def _normalize_term(term):
    """Lowercase a term and strip punctuation, preserving apostrophe-contracted words."""
    # Remove punctuation except apostrophes, then strip leading/trailing apostrophes
    cleaned = term.lower()
    cleaned = "".join(
        ch for ch in cleaned if ch not in string.punctuation or ch == "'"
    )
    cleaned = cleaned.strip("'")
    return cleaned


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
