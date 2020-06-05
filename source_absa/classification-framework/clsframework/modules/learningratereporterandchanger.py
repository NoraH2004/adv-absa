from statistics import mean

from clsframework import Callback


class LearningRateReporterAndChanger(Callback):
    """Registers the current learningrate of a registered optimizer and changes
    the learningrate of the optimizer if the registered learningrate changes
    """
    def __init__(self):
        super().__init__()
        # Register value update callbacks
        self.val_callbacks["lr"] = self.lr_change_callback

    def on_model_load(self, **kwargs):
        pass

    def on_inference_begin(self, optimizer=None, **kwargs):
        if optimizer is not None:
            meanlr = mean(g["lr"] for g in optimizer.param_groups)
            if meanlr is not None:
                return {"lr": meanlr}
        else:
            print("Warning! LearningRateReportAndChange in Stack without an optimizer! This is probably a mistake")

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

    def on_inference_end(self, **kwargs):
        pass

    def on_model_save(self, **kwargs):
        pass

    def lr_change_callback(self, lr, optimizer, **kwargs):
        if optimizer is not None:
            for g in optimizer.param_groups:
                g["lr"] = lr
