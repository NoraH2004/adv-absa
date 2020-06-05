import spacy
import re
from functools import lru_cache


class SentenceSegmentation(object):

    def __init__(self, langcode=None):
        if langcode is None:

            self.nlp_en = spacy.load("en_core_web_sm")
            self.nlp_de = spacy.load("de_core_news_sm")
            self.nlp_fr = spacy.load("fr_core_news_sm")
        if langcode == 'de':
            self.nlp_en = spacy.load("de_core_news_sm")
        elif langcode == 'fr':
            self.nlp_fr = spacy.load("fr_core_news_sm")
        elif type(langcode) == str:
            self.nlp_en = spacy.load("en_core_web_sm")

    @lru_cache(maxsize=5000)
    def split(self, document, langcode, sanitize_doc=False):
        if document is None:
            return {'text': document, 'segments': None}

        if sanitize_doc:
            # remove double whitespace
            document = str(document).strip()
            document = re.sub(r"\r\n", " ", document)
            document = re.sub(' +', ' ', document)

        if langcode == 'de':
            doc = self.nlp_de(document, disable=['tagger', 'ner'])
        elif langcode == 'fr':
            doc = self.nlp_fr(document, disable=['tagger', 'ner'])
        else:
            doc = self.nlp_en(document, disable=['tagger', 'ner'])

        return {'text': document,
                'segments': [{'text': sent.text, 'span': [sent.start_char, sent.start_char + len(sent.text)]} for
                             sent in doc.sents]}