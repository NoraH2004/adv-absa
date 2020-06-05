import os
import tempfile
import warnings

import torch

from clsframework import Callback


class BestModelKeeper(Callback):
    """Saves the model that had the best evaluation result and restores it
    on inference end
    """
    def __init__(self, metricname="accuracy", maximize=True, verbose=False, cleanup=True):
        super().__init__()
        self.metricname = metricname
        self.maximize = maximize
        self.verbose = verbose
        self.cleanup = cleanup
        self.best_model_filename = None
        self.best_model_metricval = None
        # Register value update callbacks
        self.val_callbacks[self.metricname] = self.on_metric_change

    def on_model_load(self, **kwargs):
        pass

    def on_inference_begin(self, **kwargs):
        pass

    def on_epoch_begin(self, **kwargs):
        pass

    def on_batch_begin(self, **kwargs):
        pass

    def after_forward_pass(self, **kwargs):
        pass

    def on_batch_end(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_inference_end(self, model, **kwargs):
        model = self._restore_best_model(model)
        self._remove_temp_file()
        return {"model": model}

    def on_model_save(self, **kwargs):
        pass

    def _remember_model(self, model):
        # Get random filename if necessary
        if self.best_model_filename is None:
            self.best_model_filename = tempfile.mktemp()
        # Save model
        torch.save(model.state_dict(), self.best_model_filename)

    def _restore_best_model(self, model):
        if self.verbose:
            print(
                f"BestModelKeeper: Restoring best model with metric = {self.best_model_metricval}"
            )
        if self.best_model_filename is None:
            warnings.warn(
                "BestModelKeeper: No best model was saved. Leaving model unchanged"
            )
            return model
        else:
            model.load_state_dict(torch.load(self.best_model_filename))
            return model

    def _remove_temp_file(self):
        if self.best_model_filename is not None and self.cleanup:
            os.remove(self.best_model_filename)
            self.best_model_filename = None

    def on_metric_change(self, model, **kwargs):
        metricval = kwargs[self.metricname]
        if not self.maximize:
            metricval = -metricval
        if self.best_model_metricval is None or metricval > self.best_model_metricval:
            self._remember_model(model)
            self.best_model_metricval = metricval
            if self.verbose:
                print(
                    f"BestModelKeeper: Kept new best model with metric = {metricval}"
                )

    def __del__(self):
        self._remove_temp_file()
