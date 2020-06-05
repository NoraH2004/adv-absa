from absa import train, evaluate
import unittest
import os
import shutil
import json
from absa.tests.gpu_is_available import gpu_is_available
from pprint import pprint


class TestHotelNano(unittest.TestCase):

    def setUp(self):
        self.current_file_dir = os.path.dirname(__file__)
        self.product_key = os.environ['DO_PRODUCT_KEY']

    def tearDown(self):
        if os.path.exists(self.current_file_dir + "/en-hotels-nano-absa"):
            shutil.rmtree(self.current_file_dir + "/en-hotels-nano-absa")

    @gpu_is_available
    def test_hotel_nano(self):
        # Training
        annotated_documents = []
        aspects = []
        token = self.product_key
        with open(self.current_file_dir + "/hotel_nano.jsonl") as f:
            lines = f.readlines()
        [annotated_documents.append(json.loads(line)) for line in lines]
        with open(self.current_file_dir + "/aspects.jsonl") as f:
            lines = f.readlines()
        [aspects.append(json.loads(line)['name']) for line in lines]

        def test_callback(progress=None, score=None):
            if progress is not None:
                print(f"Progress: {progress}")
            if score is not None:
                print("#############################################################")
                print(f"Score: {score}")
                print("#############################################################")

        # Docs for evaluation
        docs = [{"text": "The room was very clean. The room was very dirty. Steve was very forthcoming.",
                 "aspect_sentiments": [{"aspect": "Cleanliness", "sentiment": "POS", 'text': 'The room was very clean'},
                                       {"aspect": "Cleanliness", "sentiment": "NEG", 'text': 'The room was very dirty'},
                                       {"aspect": "Staff", "sentiment": "POS", 'text': 'Steve was very forthcoming.'}]},
                {"text": "Hugo was slow to bring the food",
                 "aspect_sentiments": [
                     {"aspect": "Staff", "sentiment": "NEG", 'text': 'Hugo was slow to bring the food'}]}

                ]
        # Start training
        train(model_folder='bert-base-uncased',
              documents=annotated_documents,
              validation_documents=docs,
              aspects=aspects,
              target=self.current_file_dir + "/en-hotels-nano-absa",
              token=token,
              batchsize=32,
              state_callback=test_callback,
              epochs=5)

        metrics = evaluate(model_folder=self.current_file_dir + "/en-hotels-nano-absa",
                           documents=docs,
                           token=token)
        self.assertLess(0.85, metrics['combined']['weighted_accuracy'])
        pprint(metrics)
