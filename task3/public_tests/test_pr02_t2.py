import unittest
from document import Document
import test_wrapper

class TestTask1DownloadSplit(unittest.TestCase):
    def test_aesop_fables(self):
        url = "https://www.gutenberg.org/files/21/21-0.txt"
        author = "Aesop"
        origin = "Aesop’s Fables"
        start_line = 848
        end_line = 5942

        import re
        search_pattern = re.compile(r'([^\n]+)\n\n(.*?)(?=\n{5}(?=[^\n]+\n\n)|$)', re.DOTALL)

        docs = test_wrapper.load_documents_from_url(url, author, origin, start_line, end_line, search_pattern)
        self.assertIsInstance(docs, list)
        self.assertGreater(len(docs), 5)
        for i, doc in enumerate(docs):
            self.assertIsInstance(doc, Document)
            self.assertEqual(doc.author, author)
            self.assertEqual(origin, doc.origin)

            if i == 1:
                self.assertEqual(doc.title.strip().lower(), "The Wolf And The Lamb".strip().lower())

            if i == len(docs) - 1:
                self.assertEqual(doc.title.strip().lower(), "The Frogs’ Complaint Against the Sun".strip().lower())

    def test_grimm_fairy_tales(self):
        url = "https://www.gutenberg.org/files/2591/2591-0.txt"
        author = "Jacob and Wilhelm Grimm"
        origin = "Grimms' Fairy Tales"
        start_line = 123
        end_line = 9239

        import re
        search_pattern = re.compile(r"([A-Z0-9 ,.'!?-]+)\n{3}(.*?)(?=\n{5}|$)", re.DOTALL)

        docs = test_wrapper.load_documents_from_url(url, author, origin, start_line, end_line, search_pattern)
        self.assertIsInstance(docs, list)
        self.assertGreater(len(docs), 5)
        for i, doc in enumerate(docs):
            self.assertIsInstance(doc, Document)
            self.assertEqual(doc.author, author)
            self.assertEqual(origin, doc.origin)

            if i == 0:
                self.assertEqual(doc.title.strip().lower(), "THE GOLDEN BIRD".strip().lower())

            if i == len(docs) - 1:
                self.assertEqual(doc.title.strip().lower(), "SNOW-WHITE AND ROSE-RED".strip().lower())

