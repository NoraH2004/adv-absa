from __future__ import absolute_import
from absa import train, Predictor
from security import Authorization
import unittest
import os
import json
import shutil
from absa.tests.gpu_is_available import gpu_is_available


@gpu_is_available
class TestEncrypt(unittest.TestCase):

    def setUp(self):
        self.current_file_dir = os.path.dirname(__file__)
        model_folder = 'xx-base'
        # Load documents
        documents = []
        with open(self.current_file_dir + '/datasets/hotel-nano/hotel_nano.jsonl') as f:
            for line in f.readlines()[0:10]:
                documents.append(json.loads(line))
        token = os.environ['DO_PRODUCT_KEY']

        # Run ABSA train function to train an encrypted and unecrypted model
        train(model_folder, documents, [], self.current_file_dir + '/xx-base-copy-encrypted', token, encrypt=True,
              epochs=1)
        train(model_folder, documents, [], self.current_file_dir + '/xx-base-copy-unencrypted', token, encrypt=False,
              epochs=1)

    def tearDown(self):
        shutil.rmtree(self.current_file_dir + "/xx-base-copy-encrypted")
        shutil.rmtree(self.current_file_dir + "/xx-base-copy-unencrypted")

    def test_absa_encrypt(self):
        # Check that the model was saved with the correct values of do_encrypted in the config
        with open(self.current_file_dir + '/xx-base-copy-encrypted/config.json') as file:
            config = json.load(file)
            assert (config['do_encrypted'] == True)

        with open(self.current_file_dir + '/xx-base-copy-unencrypted/config.json') as file:
            config = json.load(file)
            assert (config['do_encrypted'] == False)
