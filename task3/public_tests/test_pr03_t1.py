import unittest
from document import Document
from test_wrapper import stem_term, linear_boolean_search
import Levenshtein


class TestTask1Stemming(unittest.TestCase):
    def test_data_types(self):
        stemmed_term = stem_term('koala')
        self.assertIsInstance(stemmed_term, str)

    def test_boolean_search_stemmed(self):
        d1 = Document(0, "Doc1", "We need to connect devices", ["we", "need", "to", "connect", "devices"], "Author", "Origin")
        d2 = Document(1, "Doc2", "The devices are connected", ["the", "devices", "are", "connected"], "Author", "Origin")
        d3 = Document(2, "Doc3", "A secure network connection", ["a", "secure", "network", "connection"], "Author", "Origin")
        d4 = Document(3, "Doc4", "Totally unrelated content", ["totally", "unrelated", "content"], "Author", "Origin")

        # Searching for "connecting" should match d1, d2, and d3 when stemmed
        result = linear_boolean_search("connecting", [d1, d2, d3, d4], stopword_filtered=False,
                                                    stemmed=True)
        correct_ids = {0, 1, 2}
        result_ids = set([r[1].document_id for r in result])

        self.assertEqual(correct_ids, result_ids)

    def test_stemming1(self):
        test_cases = {
            'program': 'program',
            'programs': 'program',
            'programmer': 'programm',
            'programming': 'program',
            'programmers': 'programm'
        }

        for word in test_cases:
            expected_stem = test_cases[word]
            result_stem = stem_term(word)

            distance = Levenshtein.distance(expected_stem, result_stem)
            self.assertLessEqual(distance, 1, f"Strings are too different: distance = {distance}")

    def test_stemming2(self):
        test_cases = {
            "caresses": "caress",
            "ponies": "poni",
            "ties": "ti",
            "caressed": "caress",
            "hopping": "hop"
        }

        for word in test_cases:
            expected_stem = test_cases[word]
            result_stem = stem_term(word)

            distance = Levenshtein.distance(expected_stem, result_stem)
            self.assertLessEqual(distance, 1, f"Strings are too different: distance = {distance}")