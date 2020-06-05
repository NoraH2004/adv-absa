import numpy as np
from clsframework import Classifier


def test_base_model():
    cls = Classifier(model="albert-base-v2", num_labels=3, device="cpu")
    res = cls([("Hello World", "Bla"), ("The room was nice", "Cleanliness")])
    assert len(res) == 2
    res = cls([("Hello World", "Bla"), ("The room was nice", "Cleanliness")], [0, 1])
    assert isinstance(res, float)


def test_save_load():
    cls = Classifier(model="albert-base-v2", num_labels=3, device="cpu")
    res1 = cls.forward([("Hello World", "Bla"), ("The room was nice", "Cleanliness")])
    cls.save("model")
    cls = Classifier(model="model", device="cpu")
    res2 = cls.forward([("Hello World", "Bla"), ("The room was nice", "Cleanliness")])
    assert np.array_equal(res1, res2)
