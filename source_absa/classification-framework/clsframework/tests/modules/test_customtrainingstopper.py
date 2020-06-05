from clsframework import CallbackHandler, StopTrainingException
from clsframework.modules import CustomTrainingStopper
import pytest


def rettrue():
    return True


def retfalse():
    return False


class StopAfterFiveCalls:
    def __init__(self):
        self.count = 5

    def __call__(self):
        self.count -= 1
        return self.count == 0


def test_base():
    # Should raise exception
    ch = CallbackHandler([CustomTrainingStopper(stop_training_callback=rettrue)])
    with pytest.raises(StopTrainingException):
        ch.on_batch_end()
    # Should not raise exception
    ch = CallbackHandler([CustomTrainingStopper(stop_training_callback=retfalse)])
    ch.on_batch_end()


def test_classcall():
    # Should raise exception after fifth batch
    ch = CallbackHandler([CustomTrainingStopper(stop_training_callback=StopAfterFiveCalls())])
    ch.on_batch_end()
    ch.on_batch_end()
    ch.on_batch_end()
    ch.on_batch_end()
    with pytest.raises(StopTrainingException):
        ch.on_batch_end()
