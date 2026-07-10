# Contains a unified class definition for a document.
# The implementation of this class may be altered, but the original public attributes/methods are accessible.
# E. g. filtered_terms() may be changed to use to online filtering.

MAX_PREVIEW_SIZE = 10


class Document(object):
    def __init__(self, document_id=None, title="", raw_text="", terms=[], author="", origin=""):
        self.document_id = document_id  # Unique document ID
        self.title = title  # String containing the title of the document
        self.raw_text = raw_text  # String that holds the complete text of the document.
        self.terms = terms  # List of terms (strings) in the document.
        self._filtered_terms = []  # Holds terms without stopwords.
        self._stemmed_terms = []  # Holds terms that were stemmed with Porter algorithm.
        self._filtered_stemmed_terms = []  # Terms that were filtered and stemmed.
        self.author = author
        self.origin = origin

    def __str__(self):
        shortened_content = (self.raw_text[:MAX_PREVIEW_SIZE] +
                             "...") if len(self.raw_text) > MAX_PREVIEW_SIZE else self.raw_text
        return 'D' + str(self.document_id).zfill(3) + ': ' + self.title + '("' + shortened_content + '")'

    def filtered_terms(self):
        return self._filtered_terms
    
    def stemmed_terms(self):
        return self._stemmed_terms
    
    def filtered_stemmed_terms(self):
        return self.filtered_stemmed_terms
