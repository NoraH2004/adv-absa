import copy
import os
import tempfile

import torch

from clsframework import Callback, ModelListEmptyException


class ModelAverager(Callback):
    """Module that automatically averages at most the last n models. The
    models are saved on disk to limit RAM usage
    """

    def __init__(self, buffersize=1, minspacing=0, cleanup=True):
        super().__init__()
        self.buffersize = buffersize     # Max number of models to keep for averaging
        self.minspacing = minspacing     # Min number of samples between two saved models
        self.modelfilenames = []         # Buffer containing model filenames to be averaged
        self.nextadd = minspacing        # Number of samples the model should have seen at next save
        self.cleanup = cleanup           # If true, all temporary files are removed on_inference_end

    def _remember_model(self, model):
        """Adds model to list of models to be averaged over
        """
        # Get random filename
        tmpfilename = tempfile.mktemp()
        # Save model and remember filename
        torch.save(model.state_dict(), tmpfilename)
        self.modelfilenames.append(tmpfilename)
        # Remove oldest filename and file if we are above buffersize
        if len(self.modelfilenames) > self.buffersize:
            filenametoremove = self.modelfilenames[0]
            os.remove(filenametoremove)
            self.modelfilenames = self.modelfilenames[-self.buffersize:]

    def _delete_temporary_files(self):
        """Removes all temorarily created files
        """
        for filename in self.modelfilenames:
            os.remove(filename)
        self.modelfilenames = []

    def _get_average(self, model):
        """Returns the average of all saved models.
        Throws a ModelListEmptyException if the model list is empty
        """
        if len(self.modelfilenames) == 0:
            raise ModelListEmptyException("Model list was empty when calling get_average on ModelAverager class!")

        # Copy a model to work with
        tmpmodel = copy.deepcopy(model).to("cpu")
        modelparams = []

        # Get a list of all weights of all saved models
        for filename in self.modelfilenames:
            tmpmodel.load_state_dict(torch.load(filename))
            modelparams.append(copy.deepcopy(dict(tmpmodel.named_parameters())))

        # Average all weights of the model
        avgparams = {}
        for key in list(modelparams[0].keys()):
            avgparams[key] = sum(d[key] for d in modelparams) / len(modelparams)

        # Load average weights into copied model and return
        model.load_state_dict(avgparams)
        return model

    def on_inference_begin(self, **kwargs):
        pass

    def on_epoch_begin(self, **kwargs):
        pass

    def on_batch_begin(self, **kwargs):
        pass

    def after_forward_pass(self, model, nbseensamples, summarywriter=None, **kwargs):
        if nbseensamples >= self.nextadd:
            self.nextadd = nbseensamples + self.minspacing
            self._remember_model(model)
            if summarywriter is not None:
                summarywriter.add_text("ModelAverager", "Model saved for averaging", nbseensamples)

    def on_batch_end(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    # We do the averaging on end of inference and not on_model_save
    # since we might want to test a model after training without
    # having to save and load the model
    def on_inference_end(self, model, summarywriter=None, nbseensamples=None, **kwargs):
        if summarywriter is not None:
            summarywriter.add_text("ModelAverager",
                                   f"Average created over {len(self.modelfilenames)} models", nbseensamples)
        avg_model = self._get_average(model)
        if self.cleanup:
            self._delete_temporary_files()
        return {"model": avg_model}

    def on_model_save(self, **kwargs):
        pass
