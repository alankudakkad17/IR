import unittest
from document import Document
import test_wrapper

class TestTask3BooleanSearch(unittest.TestCase):
    def test_boolean_search_unfiltered_basic(self):
        d1 = Document(0, "Doc1", "alpha beta gamma", ["alpha", "beta", "gamma"], "Author", "Origin")
        d2 = Document(1, "Doc2", "delta epsilon", ["delta", "epsilon"], "Author", "Origin")
        result = test_wrapper.linear_boolean_search("beta", [d1, d2], stopword_filtered=False)
        self.assertEqual(result, [(1, d1), (0, d2)])

    def test_boolean_search_filtered_match(self):
        d1 = Document(0, "Doc1", "alpha beta gamma", ["alpha", "beta", "gamma"], "Author", "Origin")
        d2 = Document(1, "Doc2", "delta epsilon", ["delta", "epsilon"], "Author", "Origin")
        d1._filtered_terms = ["alpha", "gamma"]
        d2._filtered_terms = ["delta", "epsilon"]
        result = test_wrapper.linear_boolean_search("beta", [d1, d2], stopword_filtered=True)
        self.assertEqual(result, [(0, d1), (0, d2)])

    def test_boolean_search_case_insensitive(self):
        d1 = Document(0, "Doc1", "Alpha BETA Gamma", ["Alpha", "BETA", "Gamma"], "Author", "Origin")
        d2 = Document(1, "Doc2", "delta epsilon", ["delta", "epsilon"], "Author", "Origin")
        result = test_wrapper.linear_boolean_search("beta", [d1, d2], stopword_filtered=False)
        self.assertEqual(result, [(1, d1), (0, d2)])
