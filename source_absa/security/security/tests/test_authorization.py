from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import time
import jwt
from security import Authorization
import os
from jwt.exceptions import ExpiredSignatureError


class TestSecurity(unittest.TestCase):

    def test_authorize(self):
        key = os.environ['DO_KEY']
        valid_exp = int(time.time() + 86400)  # +1 day
        invalid_exp = int(time.time() - 86400)  # -1 day
        valid_token = jwt.encode({'exp': valid_exp}, key, algorithm='RS256').decode('utf-8')
        invalid_token = jwt.encode({'exp': invalid_exp}, key, algorithm='RS256').decode('utf-8')
        public_key = Authorization.public_key()
        self.assertTrue(isinstance(jwt.decode(valid_token, public_key, algorithms=['RS256']), dict))
        exception = False
        try:
            jwt.decode(invalid_token, public_key, algorithms=['RS256'])
        except ExpiredSignatureError:
            exception = True
        self.assertTrue(exception)

    def test_token_languages(self):
        # Check that generated token is authorized
        languages = ['en', 'de', 'fr']
        token = Authorization.generate_token('test', 1, key=os.environ['DO_KEY'], languages=languages)
        self.assertEqual(Authorization.authorize(token)['info']['languages'], languages)
        self.assertTrue(Authorization.authorize(token, language='fr')['verified'])

    def test_asterisk_languages(self):
        # Check that generated token is authorized
        languages = ['*']
        token = Authorization.generate_token('test', 1, key=os.environ['DO_KEY'], languages=languages)
        self.assertEqual(Authorization.authorize(token)['info']['languages'], languages)
        self.assertTrue(Authorization.authorize(token, language='fr')['verified'])
        self.assertTrue(Authorization.authorize(token, language='de')['verified'])
        self.assertTrue(Authorization.authorize(token, language='no_real_lang_code')['verified'])

    def test_functionality(self):
        # Check token functionality authorization
        token = Authorization.generate_token('test', 1, key=os.environ['DO_KEY'], functionality=['Analysis/*'])
        self.assertTrue(Authorization.authorize(token, functionality='Analysis/Aspect-Sentiments')['verified'])
        self.assertFalse(Authorization.authorize(token, functionality='Train/Base')['verified'])

    def test_asterisk_funcitonality(self):
        token = Authorization.generate_token('test', 1, key=os.environ['DO_KEY'], functionality=['absa/*'])
        self.assertTrue(Authorization.authorize(token, functionality='absa/predict')['verified'])
        self.assertFalse(Authorization.authorize(token, functionality='someNotAbsaFunctionality')['verified'])

    def test_double_asterisk_functionality(self):
        token = Authorization.generate_token('test', 1, key=os.environ['DO_KEY'], functionality=['*/*'], languages=['en'])
        self.assertTrue(Authorization.authorize(token, functionality='Analysis/Aspect-Sentiments')['verified'])
        self.assertTrue(Authorization.authorize(token, functionality='DeepOpinion/*')['verified'])
        self.assertTrue(Authorization.authorize(token, functionality='someweirdstuff')['verified'])