import warnings

import numpy as np
import torch

import clsframework.utils as clsu
from clsframework import Classifier


def test_numpy_fixed_seed():
    assert not np.alltrue(np.random.rand(3, 3) == np.random.rand(3, 3))
    clsu.fix_seed(23)
    arr1 = np.random.rand(3, 3)
    clsu.fix_seed(23)
    arr2 = np.random.rand(3, 3)
    assert np.alltrue(arr1 == arr2)


def test_torch_fixed_seed():
    assert not torch.equal(torch.rand(3, 3), torch.rand(3, 3))
    clsu.fix_seed(23)
    arr1 = torch.rand(3, 3)
    clsu.fix_seed(23)
    arr2 = torch.rand(3, 3)
    assert torch.equal(arr1, arr2)


def test_classifier_fixed_seed_cpu():
    t1 = Classifier(model="albert-base-v2", device="cpu").model.classifier.weight.detach()
    t2 = Classifier(model="albert-base-v2", device="cpu").model.classifier.weight.detach()
    assert not torch.equal(t1, t2)
    clsu.fix_seed(23)
    t1 = Classifier(model="albert-base-v2", device="cpu").model.classifier.weight.detach()
    clsu.fix_seed(23)
    t2 = Classifier(model="albert-base-v2", device="cpu").model.classifier.weight.detach()
    assert torch.equal(t1, t2)


def test_classifier_fixed_seed_cuda():
    if torch.cuda.is_available():
        t1 = Classifier(model="albert-base-v2", device="cuda").model.classifier.weight.detach()
        t2 = Classifier(model="albert-base-v2", device="cuda").model.classifier.weight.detach()
        assert not torch.equal(t1, t2)
        clsu.fix_seed(23)
        t1 = Classifier(model="albert-base-v2", device="cuda").model.classifier.weight.detach()
        clsu.fix_seed(23)
        t2 = Classifier(model="albert-base-v2", device="cuda").model.classifier.weight.detach()
        assert torch.equal(t1, t2)
    else:
        warnings.warn("test_classifier_fixed_seed_cuda could not be run since no CUDA device is available")
