"""
Information Retrieval System - Text-based User Interface
"""

import re
import os
from my_module import (
    load_collection_from_url,
    linear_boolean_search,
    vector_space_search,
    remove_stop_words,
    remove_stop_words_by_frequency,
    precision_recall,
)
from document import Document

# Global document collection
collection: list[Document] = []
global_ground_truth: dict[str, set[int]] = {}


def cmd_load_collection():
    """Prompt the user for collection parameters and load documents from a URL."""
    url = input("Enter URL (.txt file): ").strip()
    while not url:
        url = input("URL cannot be empty. Please enter a valid URL: ").strip()
    author = input("Author name: ").strip()
    origin = input("Collection title (origin): ").strip()
    try:
        start_line = int(input("Start line (1-based): ").strip())
        end_line = int(input("End line (1-based): ").strip())
    except ValueError:
        print("Invalid line numbers.")
        return

    pattern_str = input("Regex search pattern (leave blank for default Aesop pattern): ").strip()
    if not pattern_str:
        pattern_str = r'([^\n]+)\n\n(.*?)(?=\n{5}(?=[^\n]+\n\n)|$)'
    try:
        search_pattern = re.compile(pattern_str, re.DOTALL)
    except re.error as e:
        print(f"Invalid regex pattern: {e}")
        return

    print(f"Downloading from {url} ...")
    global collection
    try:
        collection = load_collection_from_url(url, search_pattern, start_line, end_line, author, origin)
        print(f"Loaded {len(collection)} documents.")
    except Exception as e:
        print(f"\nError: Could not load the collection. Details: {e}")


def cmd_load_ground_truth():
    """Load ground truth from a text file."""
    filepath = input("Path to ground truth file (e.g. ground_truth.txt): ").strip()
    if not os.path.isfile(filepath):
        print("File not found.")
        return

    global global_ground_truth
    global_ground_truth.clear()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ' - ' in line:
                    term, ids_str = line.split(' - ', 1)
                    term = term.strip().lower()
                    ids = set()
                    for i in ids_str.split(','):
                        i = i.strip()
                        if i.isdigit():
                            ids.add(int(i) - 1)  # Convert to 0-based
                    global_ground_truth[term] = ids
        print(f"Loaded ground truth for {len(global_ground_truth)} term(s).")
    except Exception as e:
        print(f"Error loading ground truth: {e}")


def evaluate_and_print(query: str, retrieved_docs: list[Document]):
    """Evaluate and print precision and recall if ground truth is available."""
    if not global_ground_truth:
        return
        
    query_terms = query.lower().split()
    if not query_terms:
        return
        
    # Get relevant documents for the query (intersection of individual term relevant sets)
    relevant = None
    for term in query_terms:
        if term not in global_ground_truth:
            print(f"Precision: -1.0")
            print(f"Recall: -1.0")
            print(f"(Evaluation not possible: '{term}' not in ground truth)")
            return
        term_relevant = global_ground_truth[term]
        if relevant is None:
            relevant = set(term_relevant)
        else:
            relevant.intersection_update(term_relevant)

    retrieved = {doc.document_id for doc in retrieved_docs}
    p, r = precision_recall(retrieved, relevant)
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")


def cmd_search():
    """Search the collection for a single keyword using Boolean retrieval."""
    if not collection:
        print("No documents loaded. Please load a collection first.")
        return

    term = input("Enter search term: ").strip()
    use_filtered = input("Use stop-word-filtered terms? (y/n): ").strip().lower() == "y"
    use_stemming = input("Enable stemming? (y/n): ").strip().lower() == "y"

    results = linear_boolean_search(term, collection, stopword_filtered=use_filtered, stemmed=use_stemming)
    matching = [(score, doc) for score, doc in results if score > 0]

    if not matching:
        print("No documents found.")
    else:
        print(f"Found {len(matching)} document(s):")
        for score, doc in matching:
            print(f"  [{score}] {doc}")
            
    evaluate_and_print(term, [doc for _, doc in matching])


def cmd_vsm_search():
    """Search the collection using Vector Space Model."""
    if not collection:
        print("No documents loaded. Please load a collection first.")
        return

    query = input("Enter search query (terms separated by spaces): ").strip()
    use_filtered = input("Use stop-word-filtered terms? (y/n): ").strip().lower() == "y"
    use_stemming = input("Enable stemming? (y/n): ").strip().lower() == "y"

    results = vector_space_search(query, collection, stopword_filtered=use_filtered, stemmed=use_stemming)
    # Sort results by descending score
    results = sorted(results, key=lambda x: -x[0])
    matching = [(score, doc) for score, doc in results if score > 0]

    if not matching:
        print("No documents found.")
    else:
        print(f"Found {len(matching)} document(s):")
        for score, doc in matching:
            print(f"  [{score:.4f}] {doc}")
            
    evaluate_and_print(query, [doc for _, doc in matching])


def cmd_apply_stopwords_list():
    """Apply list-based stop word removal to all documents in the collection."""
    if not collection:
        print("No documents loaded.")
        return

    filepath = input("Path to stop word file: ").strip()
    if not os.path.isfile(filepath):
        print("File not found.")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        stopwords = {line.strip().replace(" ", "") for line in f}

    for doc in collection:
        doc._filtered_terms = remove_stop_words(doc.terms, stopwords)

    print(f"Stop word filtering applied to {len(collection)} documents.")


def cmd_apply_stopwords_frequency():
    """Apply frequency-based stop word removal to all documents in the collection."""
    if not collection:
        print("No documents loaded.")
        return

    try:
        common = float(input("Common frequency threshold (e.g. 0.9): ").strip())
        rare = float(input("Rare frequency threshold (e.g. 0.1): ").strip())
    except ValueError:
        print("Invalid frequency value.")
        return

    for doc in collection:
        doc._filtered_terms = remove_stop_words_by_frequency(doc.terms, collection, low_freq=rare, high_freq=common)

    print(f"Frequency-based stop word filtering applied to {len(collection)} documents.")


def print_menu():
    print("\n=== Information Retrieval System ===")
    print("1) Load document collection from URL")
    print("2) Search (Boolean linear search)")
    print("3) Apply stop word removal (list-based)")
    print("4) Apply stop word removal (frequency-based)")
    print("5) Search (Vector Space Model)")
    print("6) Load ground truth file")
    print("0) Exit")


def main():
    while True:
        print_menu()
        choice = input("Choose an option: ").strip()
        if choice == "1":
            cmd_load_collection()
        elif choice == "2":
            cmd_search()
        elif choice == "3":
            cmd_apply_stopwords_list()
        elif choice == "4":
            cmd_apply_stopwords_frequency()
        elif choice == "5":
            cmd_vsm_search()
        elif choice == "6":
            cmd_load_ground_truth()
        elif choice == "0":
            print("Goodbye...")
            break
        else:
            print("Invalid selection. Please try again.")


if __name__ == "__main__":
    main()
