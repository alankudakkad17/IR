import unittest
from test_wrapper import precision_recall


class TestTask3Evaluation(unittest.TestCase):
    def test_all_correct(self):
        retrieved = {1, 2, 3}
        relevant = {1, 2, 3}
        precision, recall = precision_recall(retrieved, relevant)
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)

    def test_some_correct(self):
        retrieved = {1, 4, 5}
        relevant = {1, 2, 3}
        precision, recall = precision_recall(retrieved, relevant)
        self.assertAlmostEqual(precision, 1 / 3)
        self.assertAlmostEqual(recall, 1 / 3)

    def test_none_correct(self):
        retrieved = {4, 5, 6}
        relevant = {1, 2, 3}
        precision, recall = precision_recall(retrieved, relevant)
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)

    def test_no_relevant_documents(self):
        retrieved = {1, 2, 3}
        relevant = set()
        precision, recall = precision_recall(retrieved, relevant)
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)

    def test_no_retrieved_documents(self):
        retrieved = set()
        relevant = {1, 2, 3}
        precision, recall = precision_recall(retrieved, relevant)
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)

    def test_empty_inputs(self):
        retrieved = set()
        relevant = set()
        precision, recall = precision_recall(retrieved, relevant)
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)
