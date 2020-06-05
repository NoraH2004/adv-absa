import unittest
from nlp import SentenceSegmentation
from nlp import LanguageDetection
from nlp import Translation


class TestNlp(unittest.TestCase):

    def test_split_document(self):
        # Speedup through faster initialization, only loads english spacy model
        ss = SentenceSegmentation(langcode='en')
        document = "This is a  sentence. This is the second sentence. \n"
        result = ss.split(document, 'en', sanitize_doc=True)
        self.assertEqual(2, len(result))
        self.assertEqual("This is a sentence.", result['segments'][0]['text'])
        self.assertEqual("This is the second sentence.", result['segments'][1]['text'])
        self.assertSequenceEqual([0, 19], result['segments'][0]['span'])
        self.assertSequenceEqual([20, 48], result['segments'][1]['span'])

    def test_language_detection(self):
        ld = LanguageDetection()
        doc_en = "This is an English sentence!"
        doc_de = "Das ist ein deutscher Satz"
        self.assertEqual('en', ld.detect(doc_en))
        self.assertEqual('de', ld.detect(doc_de))

    def test_translation(self):
        t = Translation()
        docs = ["Das Pferd isst keinen Gurkensalat. Das sagte einst Graham Bell, als der das Telefon erfand!",
                "Alan Turing entzifferte Enigma."]
        result = t.translate(documents=docs, target='en')
        print(result)

    def test_language_detection_fallback(self):
        ld = LanguageDetection()
        doc = ":)"
        self.assertEqual(None, ld.detect(doc))
