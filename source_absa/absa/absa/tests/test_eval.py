from __future__ import absolute_import
import unittest
from absa import evaluate
import json
import os


class TestEval(unittest.TestCase):

    def setUp(self):
        self.product_key = os.environ['DO_PRODUCT_KEY']

    def load_hotel_dataset(self):
        # load 20 documents from hotel dataset
        self.current_file_dir = os.path.dirname(__file__) + '/datasets/hotel'
        annotated_documents = []
        aspects = []

        with open(self.current_file_dir + "/hotel.jsonl") as f:
            lines = f.readlines()
        [annotated_documents.append(json.loads(line)) for line in lines[0:20]]
        with open(self.current_file_dir + "/aspects.jsonl") as f:
            lines = f.readlines()
        [aspects.append(json.loads(line)['name']) for line in lines]
        return annotated_documents, aspects

    def test_eval(self):
        token = self.product_key
        annotated_hotel_documents, allaspects = self.load_hotel_dataset()

        metrics = evaluate(model_folder='en-hotels-absa',
                               documents=annotated_hotel_documents,
                               token=token)

        self.assertSetEqual({'aspect', 'aspect_details', 'sentiment', 'combined'}, set(metrics.keys()))
        self.assertEqual(0.96, metrics['combined']['weighted_accuracy'])
