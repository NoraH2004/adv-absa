from clsframework import Callback
from time import time
from clsframework.callbackhandler import CallbackHandler
import torch
import numpy as np


class ScoreReporter(Callback):
    """ Evaluates the score with a given evaluator and calls an external callback to report it back
    How often evaluation is done, depends on the the validation set you give and the percentage of time
    the validation could take.
     """

    def __init__(self, reporter, evaluator, X, Y, max_eval_percentage=10):
        """
        Parameters
        ----------
        reporter : callback
            Callback function that will be called with the param score to report the score
        evaluator : callback
            Callback function that will be called with Y_true and Y_pred. It is used to calculcate the score
        X : List
            List of tupels
        Y : List
            List true labels
        max_eval_percentage : int
            The maximum time the evaluation time is allowed to use compared to the overall execution time
        """
        super().__init__()
        self.max_eval_percentage = max_eval_percentage
        self.batch_start = 0
        self.batch_end = 0
        self.eval_durations = []
        self.avg_eval_duration = 1
        self.batch_count = 0
        self.batch_durations = []
        self.avg_batch_duration = None
        self.X = X
        self.Y = Y
        self.score = {}
        self.val_loss = None
        self.reporter = reporter
        self.evaluator = evaluator
        self.ch = CallbackHandler([])
        self.interval = 10
        self.loss_fct = torch.nn.NLLLoss()  # Used to calculate validation loss

    def evaluate(self, classifier):
        eval_start = time()
        # Switch CallbackHandler so we can evaluate in peace
        chtmp = classifier.ch
        classifier.ch = self.ch
        classifier.model.eval()
        with torch.no_grad():
            result = classifier.forward(X=self.X)  # Get class probabilities
        Y_pred = np.argmax(result, axis=1)         # Get predicted classes
        classifier.model.train()
        classifier.ch = chtmp
        # Call given evaluator
        self.score = self.evaluator(Y_true=self.Y, Y_pred=Y_pred)
        self.val_loss = self.loss_fct(torch.tensor(result), torch.tensor(self.Y)).item()
        # Call external callback
        if self.reporter is not None:
            self.reporter(score=self.score)
        eval_end = time()
        self.eval_durations.append(eval_end - eval_start)
        self.avg_eval_duration = sum(self.eval_durations) / len(self.eval_durations)

    def on_batch_begin(self, **kwargs):
        if self.batch_count < 3:
            self.batch_start = time()

    def on_batch_end(self, classifier, **kwargs):
        # Calculates start params to find the right interval
        if self.batch_count < 3:
            self.evaluate(classifier=classifier)
            self.batch_end = time()
            self.batch_durations.append(self.batch_end - self.batch_start)
            self.avg_batch_duration = sum(self.batch_durations) / len(self.batch_durations)
            eval_batch_percentage = 100 * self.avg_batch_duration / self.avg_batch_duration
            min_interval = 10
            self.interval = round(max(min_interval, eval_batch_percentage / self.max_eval_percentage))
        # Evaluates every interval
        elif self.batch_count % self.interval == 0:
            self.evaluate(classifier=classifier)
            # Write back to notify other modules
            return {'val_loss': self.val_loss}
        self.batch_count += 1
