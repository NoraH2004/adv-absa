from langdetect import detect
from functools import lru_cache


class LanguageDetection(object):

    # Returns the 639-1 code (2 digit code) of the detected language, or None if language can not be detected
    @lru_cache(maxsize=5000)
    def detect(self, document):
        try:
            language_code = detect(document)
        except:
            language_code = None
        return language_code