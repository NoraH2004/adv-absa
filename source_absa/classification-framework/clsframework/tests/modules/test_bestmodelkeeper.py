import os

import torch

from clsframework.modules import BestModelKeeper


def createmodel(fillval):
    model = torch.nn.Linear(20, 1)
    t = torch.zeros_like(model.weight.data)
    torch.fill_(t, fillval)
    model.weight.data = t
    model.bias.data = torch.tensor([fillval])
    return model


def test_best_model_keeper():
    modelkeep = BestModelKeeper(metricname="val_accuracy", maximize=True, verbose=True, cleanup=True)
    modelkeep.on_metric_change(model=createmodel(float(1)), val_accuracy=0.1)
    filename = modelkeep.best_model_filename
    modelkeep.on_metric_change(model=createmodel(float(2)), val_accuracy=0.2)
    modelkeep.on_metric_change(model=createmodel(float(3)), val_accuracy=0.3)
    modelkeep.on_metric_change(model=createmodel(float(4)), val_accuracy=0.4)
    modelkeep.on_metric_change(model=createmodel(float(1)), val_accuracy=0.1)
    model = modelkeep.on_inference_end(model=createmodel(float(1)))["model"]
    assert model.weight.data[0][0].item() == 4
    assert os.path.isfile(filename) is False

    modelkeep = BestModelKeeper(metricname="val_accuracy", maximize=False, verbose=True, cleanup=True)
    modelkeep.on_metric_change(model=createmodel(float(1)), val_accuracy=1)
    filename = modelkeep.best_model_filename
    modelkeep.on_metric_change(model=createmodel(float(2)), val_accuracy=0.5)
    modelkeep.on_metric_change(model=createmodel(float(3)), val_accuracy=0.25)
    modelkeep.on_metric_change(model=createmodel(float(4)), val_accuracy=0.125)
    modelkeep.on_metric_change(model=createmodel(float(1)), val_accuracy=0.24)
    model = modelkeep.on_inference_end(model=createmodel(float(1)))["model"]
    assert model.weight.data[0][0].item() == 4
    assert os.path.isfile(filename) is False


def test_empty_model_keeper():
    modelkeep = BestModelKeeper(metricname="val_accuracy", maximize=True, verbose=True, cleanup=True)
    modelkeep._remove_temp_file()
