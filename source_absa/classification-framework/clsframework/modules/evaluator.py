import numpy as np
import torch
from torch.nn import NLLLoss

from clsframework import Callback, CallbackHandler
from clsframework.modules import SortByLength


class Evaluator(Callback):
    """Evaluates a given model in regular intervals on a validation set
    and writes the results to the screen as well as to a summarywriter
    if one is defined. The calculated accuracy is also registered in
    the global state

    """

    def __init__(self, X, Y, interval, verbose=False):
        super().__init__()
        self.X = X
        self.Y = Y
        self.interval = interval
        self.nexteval = interval
        self.ch = CallbackHandler([SortByLength()])
        self.verbose = verbose
        self.loss_fct = NLLLoss()

    def on_inference_begin(self, **kwargs):
        pass

    def on_epoch_begin(self, **kwargs):
        pass

    def on_batch_begin(self, **kwargs):
        pass

    def after_forward_pass(self, **kwargs):
        pass

    def on_batch_end(self, nbseensamples, classifier, summarywriter=None, **kwargs):
        if nbseensamples >= self.nexteval:
            self.nexteval = nbseensamples + self.interval
            # Switch CallbackHandler so we can evaluate in peace and with sorted tokens
            chtmp = classifier.ch
            classifier.ch = self.ch
            # Run inference on given data
            classifier.model.eval()
            with torch.no_grad():
                result = classifier.forward(self.X, batchsize=16, verbose=self.verbose)
            classifier.model.train()
            y = np.argmax(result, axis=1)
            # Switch back to original callback handler
            classifier.ch = chtmp
            # Calculate accuracy and loss and print on screen
            accuracy = sum(y == self.Y) / len(self.Y)
            loss = self.loss_fct(torch.tensor(result), torch.tensor(self.Y)).item()
            if self.verbose:
                print(f"Evaluation accuracy after {nbseensamples} is {accuracy}, loss = {loss}")
            # Write accuracy and loss to tensorboard if possible
            if summarywriter is not None:
                summarywriter.add_scalar("Base/Val_Accuracy", accuracy, nbseensamples)
                summarywriter.add_scalar("Base/Val_Loss", loss, nbseensamples)

            return {"val_accuracy": accuracy, "val_loss": loss}

    def on_epoch_end(self, **kwargs):
        pass

    def on_inference_end(self, **kwargs):
        pass

    def on_model_save(self, **kwargs):
        pass
