import numpy as np
from random import shuffle
from clsframework import Callback


class ShuffleData(Callback):
    """Module that shuffles incoming data and and brings the results and
    the data in the correct order again after inference
    """

    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, X, Y, **kwargs):
        # Determine shuffled indices
        self.idxs = list(range(len(X)))
        shuffle(self.idxs)
        # Shuffle data
        newX = [X[i] for i in self.idxs]
        newY = None if Y is None else [Y[i] for i in self.idxs]
        # Update
        return {"X": newX, "Y": newY}

    def on_epoch_end(self, X, Y, results, **kwargs):
        ordering_dict = {self.idxs[i]: i for i in range(len(self.idxs))}
        # Unshuffle original data
        newX = [X[ordering_dict[i]] for i in range(len(X))]
        newY = None if Y is None else [Y[ordering_dict[i]] for i in range(len(Y))]
        # Unshuffle results (if no Y is given, otherwise results are losses and can't be unshuffled)
        if Y is None:
            res = np.empty_like(results)
            for i in range(results.shape[0]):
                res[i, :] = results[ordering_dict[i], :]
        else:
            res = results
        # Update
        return {"X": newX, "Y": newY, "results": res}
