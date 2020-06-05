from clsframework import Classifier, CallbackHandler
from clsframework.modules import ModelSampler
from clsframework.modules.modelsampler import MonteCarloDropout
import numpy as np


def test_modelsampler():
    # Ensure results without ModelSampler are deterministic
    X = ["Once", "upon", "a", "midnight", "dreary", "while", "I", "pondered", "weak", "and", "weary"]
    cls = Classifier(model="albert-base-v2", num_labels=10, device="cpu")
    y1 = cls.forward(X)
    y2 = cls.forward(X)
    assert np.alltrue(y1 == y2)

    # Ensure results with ModelSampler are NOT deterministic
    X = ["ox ver", "many", "a", "quaint", "and", "curious", "volume", "of", "forgotten", "lore"]
    modelsample = ModelSampler()
    cls = Classifier(model="albert-base-v2", num_labels=10, device="cpu", ch=CallbackHandler([modelsample]))
    assert len(modelsample.dropoutlayers) == 3
    y1 = cls.forward(X)
    y2 = cls.forward(X)
    assert not np.alltrue(y1 == y2)

    # Check if dropouts have actually been changed
    for obj in modelsample.dropoutlayers:
        assert isinstance(obj, MonteCarloDropout)

    # Ensure same input still gives same output in the same epoch and
    # different inputs give different outputs
    X = ["Hello", "Test", "Hello", "Test"]
    modelsample = ModelSampler()
    cls = Classifier(model="albert-base-v2", num_labels=10, device="cpu", ch=CallbackHandler([modelsample]))
    y = cls.forward(X)
    assert np.alltrue(y[0] == y[2])
    assert np.alltrue(y[1] == y[3])
    assert not np.alltrue(y[0] == y[1])
    assert not np.alltrue(y[1] == y[2])
