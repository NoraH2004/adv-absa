import pytest
from clsframework import CallbackHandler
from clsframework import NoCallbackInstanceException


def test_empty_callbackhandler():
    ch = CallbackHandler([])
    ch.on_model_load()
    ch.on_inference_begin()
    ch.on_epoch_begin()
    ch.on_batch_begin()
    ch.after_forward_pass()
    ch.on_batch_end()
    ch.on_epoch_end()
    ch.on_inference_end()
    ch.on_model_save()
    assert len(ch.state) == 0
    assert len(ch.callbacks) == 0


def test_exceptions():
    with pytest.raises(NoCallbackInstanceException):
        CallbackHandler([23])
