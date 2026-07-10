import unittest
from document import Document
import test_wrapper

import os

test_dir = os.path.dirname(__file__)
stopword_file_path = os.path.join(test_dir, "englishST.txt")


class TestTask2StopWords(unittest.TestCase):
    def test_stopwords_from_file_basic(self):
        doc = Document(0, "Test", "The fox jumps over the lazy dog",
                       ["the", "fox", "jumps", "over", "the", "lazy", "dog"],
                       author="Test", origin="Some random collection")

        with open(stopword_file_path, "r") as f:
            stopwords = set([line.strip().replace(' ', '') for line in f])
        test_wrapper.remove_stopwords_by_list(doc, stopwords)

        self.assertEqual(doc.filtered_terms(), ["fox", "jumps", "lazy", "dog"])

    def test_stopwords_from_file_case_insensitive(self):
        doc = Document(1, "CaseTest", "The QUICK brown fox IN", ["The", "QUICK", "brown", "fox", "IN"],
                       author="Test", origin="Some random collection")
        with open(stopword_file_path, "r") as f:
            stopwords = set([line.strip().replace(' ', '') for line in f])
        test_wrapper.remove_stopwords_by_list(doc, stopwords)

        self.assertEqual(list(map(str.lower, doc.filtered_terms())), ["quick", "brown", "fox"])

    def test_stopwords_by_frequency(self):
        d1 = Document(0, "D1", "a a a a a b", ["a", "a", "a", "a", "a", "b", "e"], "Author", "Source")
        d2 = Document(1, "D2", "c d e f", ["a", "c", "d", "e", "f"], "Author", "Source")
        d3 = Document(2, "D3", "a f g d d", ["a", "f", "g", "d", "d"], "Author", "Source")
        d4 = Document(2, "D4", "a b b f", ["a", "b", "b", "f"], "Author", "Source")
        d5 = Document(2, "D5", "a b b c f", ["a", "b", "b", "c", "f"], "Author", "Source")

        test_wrapper.remove_stopwords_by_frequency(d3, [d1, d2, d3, d4, d5], rare_frequency=.1, common_frequency=.3)
        self.assertIsInstance(d3.filtered_terms(), list)
        self.assertTrue(all(isinstance(t, str) for t in d3.filtered_terms()))
        self.assertNotIn("a", d3.filtered_terms())  # a is very frequent
        self.assertNotIn("g", d3.filtered_terms())  # g is very rare
