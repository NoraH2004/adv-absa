from clsframework import Callback, StopTrainingException


class CustomTrainingStopper(Callback):
    """Module calls a stop_training_callback function after each batch and
    finishes training if the function returns True
    """
    def __init__(self, stop_training_callback):
        super().__init__()
        self.stop_training_callback = stop_training_callback
        # Register value update callbacks
        # self.val_callbacks["test"] = self.test_change_callback

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
        if self.stop_training_callback():
            raise StopTrainingException()

    def on_epoch_end(self, **kwargs):
        pass

    def on_inference_end(self, **kwargs):
        pass

    def on_model_save(self, **kwargs):
        pass
