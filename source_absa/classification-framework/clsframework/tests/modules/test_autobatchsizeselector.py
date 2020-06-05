import warnings

import os
import pytest
import torch

from clsframework import CallbackHandler, Classifier
from clsframework.modules import AutoBatchsizeSelector


# Test if we are getting a warning when using CPU as device
def test_autobatchsize_cpu():
    X = ["one", "two", "three", "four", "five", "six"]
    autobatch = AutoBatchsizeSelector(
        basemodel="albert-base-v2", maxbatchsize=32)
    ch = CallbackHandler([autobatch])
    cls = Classifier(model="albert-base-v2", device="cpu", fp16=None, ch=ch)
    with pytest.warns(UserWarning):
        cls(X)


# Warning: This test might fail if the data in
# "autobatchsizeselector.json" is changed
def test_batchsize_selection():
    autobatch = AutoBatchsizeSelector(
        basemodel="albert-base-v2", maxbatchsize=32)
    autobatch.basemodel = "albert-base-v2"
    autobatch.freemem = 6674382848
    res128 = autobatch._determine_max_batchsize(fp16="None",
                                                max_token_length=128)
    res512 = autobatch._determine_max_batchsize(fp16="None",
                                                max_token_length=512)
    assert res128 > res512
    # Test if this also works for smaller max token lengths than we
    # have data for (was a bug that occurred) See issue #23
    assert autobatch._determine_max_batchsize(fp16="None",
                                              max_token_length=5) > 5
    # Test if batchsize of 1 is correctly used when nothing useful can be found
    autobatch.freemem = 2848
    with pytest.warns(UserWarning):
        assert autobatch._determine_max_batchsize(fp16="None",
                                                  max_token_length=5) == 1
    autobatch.freemem = 6674382848
    autobatch.basemodel = "fubar"
    with pytest.warns(UserWarning):
        assert autobatch._determine_max_batchsize(fp16="None",
                                                  max_token_length=5) == 1


def test_autobatchsize_cuda():
    if torch.cuda.is_available():
        X = ["a "*200]
        autobatch = AutoBatchsizeSelector(
            basemodel="albert-base-v2", maxbatchsize=1024)
        ch = CallbackHandler([autobatch])
        cls = Classifier(model="albert-base-v2",
                         device="cuda", fp16=None, ch=ch)
        cls(X, batchsize=1024)
        assert ch["batchsize"] < 256
        # Check if maxbatchsize is respected
        autobatch = AutoBatchsizeSelector(
            basemodel="albert-base-v2", maxbatchsize=1)
        ch = CallbackHandler([autobatch])
        cls = Classifier(model="albert-base-v2",
                         device="cuda", fp16=None, ch=ch)
        cls(X, batchsize=1024)
        assert ch["batchsize"] == 1
    else:
        warnings.warn(
            "test_autobatchsize_cuda could not be run since no CUDA device is available")


def test_multi_gpu_exception():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    autobatch = AutoBatchsizeSelector(
        basemodel="albert-base-v2", maxbatchsize=1024)
    with pytest.raises(Exception) as e:
        autobatch.on_inference_begin(device="cuda")
    assert "AutoBatchsizeSelector" in str(e)
