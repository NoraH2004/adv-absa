from copy import copy

import numpy as np
from clsframework import CallbackHandler
from clsframework.modules import ShuffleData


def test_shuffledata_withlabel():
    global tokenizer
    # Prepare
    ch = CallbackHandler([ShuffleData()])
    X = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    Xorig = copy(X)
    Y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Yorig = copy(Y)
    # Let ShuffleData do its thing
    X, Y = ch.on_epoch_begin(X=X, Y=Y)
    assert X != Xorig
    assert Y != Yorig
    assert set(X) == set(Xorig)
    assert set(Y) == set(Yorig)
    # See if shuffling is reverted correctly
    results = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    X, Y, results = ch.on_epoch_end(X=X, Y=Y, results=results)
    assert X == Xorig
    assert Y == Yorig
    # With given labels results should not change
    assert results == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


def test_suffledata_withoutlabel():
    global tokenizer
    # Prepare
    ch = CallbackHandler([ShuffleData()])
    X = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    Xorig = copy(X)
    # Let SortByLength do its thing
    X, Y = ch.on_epoch_begin(X=X, Y=None)
    assert X != Xorig
    assert set(X) == set(Xorig)
    assert Y is None
    # See if results are changed (we are not checking whether the change of order was useful)
    results = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0],
                        [5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0],
                        [9.0, 9.0], [10.0, 10.0]])
    resultsorig = copy(results)
    X, Y, results = ch.on_epoch_end(X=X, Y=None, results=results)
    assert X == Xorig
    assert Y is None
    # Without labels, results should not stay the same
    assert not np.array_equal(results, resultsorig)
