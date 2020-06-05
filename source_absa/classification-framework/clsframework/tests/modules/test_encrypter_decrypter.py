from clsframework.modules import Decrypter, Encrypter
from clsframework import CallbackHandler, Classifier
import shutil


def test_train_encrypted():
    encrypted_trained_config = _get_saved_model_config(True)
    assert encrypted_trained_config.do_encrypted is True
    _cleanup()


def test_train_unencrypted():
    unencrypted_trained_config = _get_saved_model_config(False)
    assert unencrypted_trained_config.do_encrypted is False
    _cleanup()


# Load and save model with given value of encrypt
def _get_saved_model_config(encrypt):
    # Set up model
    ch = CallbackHandler([Decrypter(), Encrypter(encrypt)])
    cls = Classifier("albert-base-v2", device="cpu", ch=ch)
    # Save model
    cls.save("model")

    # Reload model and return model config
    cls = Classifier(model="model", device="cpu")
    return cls.model.config


# Cleanup function
def _cleanup():
    shutil.rmtree("model")
