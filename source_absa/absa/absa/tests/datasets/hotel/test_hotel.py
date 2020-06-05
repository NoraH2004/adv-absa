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
        if os.path.exists(self.current_file_dir + "/en-hotels-absa"):
            shutil.rmtree(self.current_file_dir + "/en-hotels-absa")

    @gpu_is_available
    def test_hotel(self):
        # Training
        annotated_documents = []
        aspects = []
        token = self.product_key
        with open(self.current_file_dir + "/hotel.jsonl") as f:
            lines = f.readlines()
        [annotated_documents.append(json.loads(line)) for line in lines]
        with open(self.current_file_dir + "/aspects.jsonl") as f:
            lines = f.readlines()
        [aspects.append(json.loads(line)['name']) for line in lines]

        # Evaluation docs
        docs = [{"text": "...",
                 "aspect_sentiments": [
                     {"aspect": "Reception", "sentiment": "NEG", "span": [70, 166],
                      "text": "We arrived about midday and there was a queue to check in, \
                      which seemed to take ages to go down."},
                     {"aspect": "Reception", "sentiment": "POS", "span": [167, 249],
                      "text": "Our check in was fairly quick but not sure why others in front of us took so long."},
                     {"aspect": "Room", "sentiment": "NEU", "span": [250, 327],
                      "text": "Our room was \"ready\" although the sofa bed for our daughter hadn't been made."},
                     {"aspect": "Bed", "sentiment": "NEG", "span": [328, 416],
                      "text": "We went out for the day and arrived back about 7pm and the sofa bed still \
                      wasn't made!!!"},
                     {"aspect": "Service", "sentiment": "NEG", "span": [417, 494],
                      "text": "This was poor service and required a phone call to reception to get it made.\n"},
                     {"aspect": "Reception", "sentiment": "NEG", "span": [417, 494],
                      "text": "This was poor service and required a phone call to reception to get it made.\n"},
                     {"aspect": "Staff", "sentiment": "NEG", "span": [417, 494],
                      "text": "This was poor service and required a phone call to reception to get it made.\n"},
                     {"aspect": "Location", "sentiment": "POS", "span": [494, 649],
                      "text": "The location is spot on, a stones throw from the Eiffel  tower and if you get the \
                      Le Bus Direct from the airport, the stop is also between tower and hotel."},
                     {"aspect": "Public transport", "sentiment": "POS",
                      "span": [494, 649],
                      "text": "The location is spot on, a stones throw from the Eiffel  tower and if you get the \
                      Le Bus Direct from the airport, the stop is also between tower and hotel."},
                     {"aspect": "Breakfast", "sentiment": "POS", "span": [650, 738],
                      "text": "Breakfast was good enough, plenty of choice so you can't go hungry \
                      unless you're fussy.\n"},
                     {"aspect": "Room", "sentiment": "NEG", "span": [738, 759],
                      "text": "Room was pretty small"},
                     {"aspect": "Hotel", "sentiment": "NEU", "span": [802, 844],
                      "text": "Overall the hotel was OK for what we paid."},
                     {"aspect": "Value for money", "sentiment": "NEU",
                      "span": [802, 844],
                      "text": "Overall the hotel was OK for what we paid."},
                     {"aspect": "Hotel", "sentiment": "POS", "span": [845, 879],
                      "text": "I'd consider stopping there again."},
                     {"aspect": "Staff", "sentiment": "NEG",
                      'text': 'Hugo was slow to bring the food'},
                     {"aspect": "Payment", "sentiment": "NEG",
                      'text': 'They did not accept my credit card.'},
                     {"aspect": "Safety", "sentiment": "NEG",
                      'text': 'I was afraid to go outside after 10pm.'},
                     {"aspect": "Value for money", "sentiment": "POS",
                      'text': 'Best hotel in town for this very small price!'}
                 ]}]

        def test_callback(progress=None, score=None):
            if progress is not None:
                print(f"Progress: {progress}")
            if score is not None:
                pass
                print("#############################################################")
                print(f"Score: {score}")
                print("#############################################################")

        train(model_folder='bert-base-uncased',
              documents=annotated_documents,
              aspects=aspects,
              target=self.current_file_dir + "/en-hotels-absa",
              token=token,
              validation_documents=docs,
              state_callback=test_callback,
              batchsize=32,
              epochs=5)

        metrics = evaluate(model_folder=self.current_file_dir + "/en-hotels-absa",
                           documents=docs,
                           token=token)
        self.assertLess(0.70, metrics['combined']['weighted_accuracy'])
        pprint(metrics)
