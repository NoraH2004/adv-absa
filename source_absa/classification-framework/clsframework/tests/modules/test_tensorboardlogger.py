import torch

from clsframework import CallbackHandler
from clsframework.modules import TensorBoardLogger


def test_logging():
    ch = CallbackHandler([TensorBoardLogger()])
    ch.on_inference_begin()
    _ = ch.on_batch_begin(x=[1, 2, 3], nbseensamples=1)
    _ = ch.after_forward_pass(y=[1, 2, 3], result=torch.tensor([1.0]),
                              nbseensamples=2, lr=0.01)
