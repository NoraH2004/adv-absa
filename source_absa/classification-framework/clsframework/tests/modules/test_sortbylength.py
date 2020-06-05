from copy import copy

import numpy as np
from clsframework import CallbackHandler
from clsframework.modules import SortByLength
from clsframework import CachedAutoTokenizer

tokenizer = CachedAutoTokenizer("albert-base-v2")


def test_sortbylength_withlabel():
    global tokenizer
    # Prepare
    ch = CallbackHandler([SortByLength()])
    X = ["a a a", "a a", "a", "a a a a"]
    Xorig = copy(X)
    Y = [3, 2, 1, 4]
    Yorig = copy(Y)
    # Let SortByLength do its thing
    X, Y, tokenizer = ch.on_epoch_begin(X=X, Y=Y, tokenizer=tokenizer)
    assert X == ["a", "a a", "a a a", "a a a a"]
    assert Y == [1, 2, 3, 4]
    # See if sorting is reverted correctly
    results = [1.0, 2.0, 3.0, 4.0]
    X, Y, results = ch.on_epoch_end(X=X, Y=Y, results=results)
    assert X == Xorig
    assert Y == Yorig
    assert results == [1.0, 2.0, 3.0, 4.0]  # With given labels results should not change


def test_sortbylength_withoutlabel():
    global tokenizer
    # Prepare
    ch = CallbackHandler([SortByLength()])
    X = ["a a a", "a a", "a", "a a a a"]
    Xorig = copy(X)
    # Let SortByLength do its thing
    X, Y, tokenizer = ch.on_epoch_begin(X=X, Y=None, tokenizer=tokenizer)
    assert X == ["a", "a a", "a a a", "a a a a"]
    # See if sorting is reverted correctly
    results = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    X, Y, results = ch.on_epoch_end(X=X, Y=None, results=results)
    assert X == Xorig
    # Without labels, results should be unsorted
    assert np.array_equal(results, np.array([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0], [4.0, 4.0]]))
