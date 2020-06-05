import torch
import os
import copy
from clsframework import CallbackHandler
from clsframework.modules import ModelAverager


def createmodel(fillval):
    model = torch.nn.Linear(20, 1)
    t = torch.zeros_like(model.weight.data)
    torch.fill_(t, fillval)
    model.weight.data = t
    model.bias.data = torch.tensor([fillval])
    return model


def test_averaging_with_cleanup():
    modavg = ModelAverager(buffersize=3, minspacing=2, cleanup=True)
    ch = CallbackHandler([modavg])
    for i in range(10):
        model = createmodel(float(i+1))
        _, _ = ch.after_forward_pass(model=model, nbseensamples=i+1)
    filelist = copy.copy(modavg.modelfilenames)
    model = ch.on_inference_end(model=model)
    assert model.weight.data[0][0].item() == 8.0
    assert model.bias.data[0].item() == 8.0
    for filename in filelist:
        assert os.path.isfile(filename) is False


def test_averaging_without_cleanup():
    modavg = ModelAverager(buffersize=3, minspacing=2, cleanup=False)
    ch = CallbackHandler([modavg])
    for i in range(10):
        model = createmodel(float(i+1))
        _, _ = ch.after_forward_pass(model=model, nbseensamples=i+1)
    filelist = copy.copy(modavg.modelfilenames)
    model = ch.on_inference_end(model=model)
    assert model.weight.data[0][0].item() == 8.0
    assert model.bias.data[0].item() == 8.0
    for filename in filelist:
        assert os.path.isfile(filename) is True
