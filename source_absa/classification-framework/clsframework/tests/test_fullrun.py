from clsframework import Classifier, CallbackHandler
from clsframework.modules import Train, TorchOptimizer, ShuffleData
import torch


def test_sentence_training():
    batchsize = 8
    X = ["zero", "one", "two", "three"] * batchsize
    Y = [0, 1, 2, 3] * batchsize
    ch = CallbackHandler([
        ShuffleData(),
        TorchOptimizer(optclass=torch.optim.Adam, lr=1e-5),
        Train()
    ])
    cls = Classifier(model="albert-base-v2", num_labels=4, device="cpu", ch=ch)
    cls(X, Y, epochs=50, batchsize=batchsize)
    cls.model.eval()
    with torch.no_grad():
        cls.ch = CallbackHandler([])
        predY = cls(["zero", "one", "two", "three"])
        assert all(predY == [0, 1, 2, 3])
