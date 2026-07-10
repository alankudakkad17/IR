import unittest
from document import Document
from test_wrapper import vector_space_search

class TestTask2VSM(unittest.TestCase):
    def test_vsm_find_relevant(self):
        d1 = Document(0, "Doc1", "the quick brown fox", ["the", "quick", "brown", "fox"], "Author", "Origin")
        d2 = Document(1, "Doc2", "jumps over the lazy dog", ["jumps", "over", "the", "lazy", "dog"], "Author", "Origin")
        d3 = Document(2, "Doc3", "completely different topic", ["completely", "different", "topic"], "Author", "Origin")

        # Query contains: "quick dog"
        result = vector_space_search("quick dog", [d1, d2, d3], stopword_filtered=False, stemmed=False)

        # Sort results by descending score (most relevant first)
        result = sorted(result, key=lambda x: -x[0])

        # Expect d1 and d2 to have a non-zero score due to shared terms
        # d1 shares "quick", d2 shares "dog", d3 has no overlap
        self.assertGreater(result[0][0], 0)
        self.assertGreater(result[1][0], 0)
        self.assertEqual(result[2][0], 0)

        # Also check that the correct documents are returned in order
        ranked_docs = [doc for score, doc in result]
        self.assertIn(d1, ranked_docs[:2])
        self.assertIn(d2, ranked_docs[:2])
        self.assertEqual(ranked_docs[2], d3)

    def test_vsm_exclude_irrelevant(self):
        d1 = Document(0, "Doc1", "data science and machine learning", ["data", "science", "and", "machine", "learning"],
                      "Author", "Origin")
        d2 = Document(1, "Doc2", "basketball and soccer", ["basketball", "and", "soccer"], "Author",
                      "Origin")  # unrelated
        d3 = Document(2, "Doc3", "data analysis using pandas", ["data", "analysis", "using", "pandas"], "Author",
                      "Origin")

        query = "machine learning with data"
        result = vector_space_search(query, [d1, d2, d3], stopword_filtered=False, stemmed=False)
        non_zero_results = [doc for score, doc in result if score > 0]

        self.assertNotIn(d2, non_zero_results)

