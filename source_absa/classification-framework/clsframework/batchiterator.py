import tqdm


class BatchIterator(object):
    """Given samples X and optionally labels Y, provides an iterator over
    all batches of X and Y with a certain batchsize

    In addition, allows for sorting of the samples in a batch given a
    sort callback and shuffeling of samples.
    """
    def __init__(self, X, Y=None, batchsize=16, verbose=False):
        super(BatchIterator, self).__init__()
        if Y is not None:
            assert len(X) == len(Y)
        self.X = X
        self.Y = Y
        self.batchsize = batchsize  # Current batchsize the iterator is providing (can change)
        self.batchsizes = []  # Number of samples for all delivered batches
        self.cursample = 0  # First sample of the next batch
        self.prevcursample = 0  # First sample of the current batch
        # Get progress bar going if verbose is true
        self.verbose = verbose
        if verbose:
            self.pbar = tqdm.tqdm(total=len(self.X))

    def repeat_batch(self, new_batchsize=None):
        """Resets the iterator so the last batch will be repeated and also
        allows to set a new batchsize.

        This is mostly intended to be able to switch to smaller batch
        sizes on the fly if we got a memory-error at some point
        """
        self.cursample = self.prevcursample
        if new_batchsize is not None:
            self.batchsize = new_batchsize
        self.batchsizes.pop(-1)

    def __iter__(self):
        return self

    def __next__(self):
        # Stop iterator if we are finished
        if self.cursample == len(self.X):
            if self.verbose:
                self.pbar.close()
            raise StopIteration
        # Construct new batch
        self.prevcursample = self.cursample
        batchend = min(self.cursample + self.batchsize, len(self.X))
        x = [self.X[i] for i in range(self.cursample, batchend)]
        y = None
        if self.Y is not None:
            y = [self.Y[i] for i in range(self.cursample, batchend)]
        # Update internal data and progress bar
        self.cursample = batchend
        self.batchsizes.append(self.batchsize)
        if self.verbose:
            self.pbar.set_description(f"Batchsize = {self.batchsize}")
            self.pbar.n = self.prevcursample
            self.pbar.refresh()

        return x, y
