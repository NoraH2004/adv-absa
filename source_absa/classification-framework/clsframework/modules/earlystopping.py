from clsframework import Callback, StopTrainingException


class EarlyStopping(Callback):
    """Module that uses information gathered from an evaluation module to
    decide when to stop training (i.e. if no improvement has been seen for
    some time)
    """
    def __init__(self,
                 patience,
                 metricname="val_accuracy",
                 maximize=True,
                 epsilon=0.001,
                 lr_reduction_patience=None,
                 lr_reduction_factor=0.5,
                 verbose=False):
        super().__init__()
        self.patience = patience  # Patience in number of seen samples
        self.metricname = metricname
        self.bestmetric = None  # Best metric that has been seen until now
        self.bestmetric_nbseensamples = 0  # Number of seen samples when best metric was observed
        self.maximize = maximize  # If False, the metric is minimized
        self.epsilon = epsilon  # Ignore improvements smaller than epsilon
        # Number of steps without improvement until lr is adapted (None = no reduction)
        self.lr_reduction_patience = lr_reduction_patience
        self.next_lr_reduction = lr_reduction_patience
        self.lr_reduction_factor = lr_reduction_factor  # Factor to reduce lr with
        self.verbose = verbose
        # Register value callbacks
        self.val_callbacks[metricname] = self.on_metric_change

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

    def on_inference_end(self, **kwargs):
        pass

    def on_model_save(self, **kwargs):
        pass

    def on_metric_change(self, nbseensamples, lr=None, **kwargs):
        res = None
        metricval = kwargs[self.metricname]
        if not self.maximize:
            metricval = -metricval

        self.verbose and print(f"{self.metricname} = {metricval}")
        # Metric improved so save new best metric and reset time to stop training
        if self.bestmetric is None or (metricval - self.epsilon > self.bestmetric):
            self.verbose and print("Metric improved!")
            self.bestmetric = metricval
            self.bestmetric_nbseensamples = nbseensamples
            if self.lr_reduction_patience is not None:
                self.next_lr_reduction = nbseensamples + self.lr_reduction_patience
        # Metric did not improve. Adapt learning rate if requested and stop training after patience
        else:
            if self.verbose:
                print(f"Early stopping at {self.bestmetric_nbseensamples + self.patience} samples without improvement.")
            # If needed, adapt learning rate
            if lr is not None and self.lr_reduction_patience is not None and nbseensamples >= self.next_lr_reduction:
                self.next_lr_reduction = nbseensamples + self.lr_reduction_patience
                new_lr = lr * self.lr_reduction_factor
                self.verbose and print(f"Reducing learning rate from {lr} to {new_lr}")
                res = {"lr": new_lr}
            # If we have exhausted patience, stop training
            if self.bestmetric_nbseensamples + self.patience <= nbseensamples:
                self.verbose and print("Stopping Early!")
                raise StopTrainingException()

            return res
