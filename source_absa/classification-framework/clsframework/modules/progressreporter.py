from clsframework import Callback


class ProgressReporter(Callback):
    """ Calculates the progress and calls a external callback if given
    """

    def __init__(self, reporter):
        """
        Parameters
        ----------
        reporter : callback
            Callback function that will be called with the param progress to report the progress
        """
        super().__init__()
        self.reporter = reporter
        self.all_samples = 0
        self.seen_samples = 0

    def on_inference_begin(self, X, epochs, **kwargs):
        self.all_samples = len(X) * epochs
        if self.reporter is not None:
            self.reporter(progress=0.00)

    def after_forward_pass(self, x, **kwargs):
        self.seen_samples += len(x)
        # Calculates progress
        progress = round(100 * self.seen_samples / self.all_samples, 2)
        # Call external callback
        if self.reporter is not None:
            self.reporter(progress=progress)
