from __future__ import absolute_import
from absa import Predictor
import unittest
import jsonschema
import os
import json
from pprint import pprint


class TestPredict(unittest.TestCase):

    def setUp(self):
        self.product_key = os.environ['DO_PRODUCT_KEY']

    def test_absa_prediction(self):
        token = self.product_key
        documents = [{'text': 'The room was tidy and well organised! The hotel lobby was dirty!'}]
        absa = Predictor(model_folder='en-hotels-absa', token=token)
        aspect_sentiments = absa.predict(documents=documents, token=token)
        print(aspect_sentiments)
        self.assertEqual('Room', aspect_sentiments[0][0]['aspect'])
        self.assertEqual('POS', aspect_sentiments[0][0]['sentiment'])
        self.assertEqual('Cleanliness', aspect_sentiments[0][1]['aspect'])
        self.assertEqual('POS', aspect_sentiments[0][1]['sentiment'])
        path = os.path.join(os.path.dirname(__file__), "../schemas/aspect_sentiments_list.json")
        with open(path) as schema_file:
            schema = json.loads(schema_file.read())
        jsonschema.validate(aspect_sentiments, schema)

    def test_absa_prediction_with_segments(self):
        token = self.product_key
        documents = [{'text': 'The room was tidy and well organised! But the staff was really unfriendly! That is what I want to say.'}]
        absa = Predictor(model_folder='en-hotels-absa', token=token)
        segments = absa.predict(documents=documents, token=token, with_segments=True)
        pprint(segments)
        path = os.path.join(os.path.dirname(__file__), "../schemas/segments_list.json")
        self.assertEqual('Room', segments[0][0]['aspect_sentiments'][0]['aspect'])
        self.assertEqual('POS', segments[0][0]['aspect_sentiments'][0]['sentiment'])
        with open(path) as schema_file:
            schema = json.loads(schema_file.read())
        jsonschema.validate(segments, schema)

    def test_absa_prediction_with_progress_reporter(self):
        token = self.product_key
        documents = [{'text': 'The room was tidy and well organised!'}] * 15
        reported_progress = 0

        def test_callback(progress=None):
            nonlocal reported_progress
            if progress is not None:
                self.assertGreaterEqual(progress, reported_progress)
                reported_progress = progress
                print(f"Progress: {reported_progress}")

        absa = Predictor(model_folder='en-hotels-absa', token=token, state_callback=test_callback)
        aspect_sentiments = absa.predict(documents=documents, token=token)
        self.assertGreaterEqual(reported_progress, 100)

    def test_absa_prediction_with_stop(self):
        token = self.product_key
        documents = [{'text': 'The room was tidy and well organised!'}] * 5

        def test_stop_callback():
            return True
        absa = Predictor(model_folder='en-hotels-absa', token=token, stop_callback=test_stop_callback)
        aspect_sentiments = absa.predict(documents=documents, token=token)
        self.assertEqual(0, len(aspect_sentiments[4]))
