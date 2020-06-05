class Callback():
    """Base class used to implement a system that is able to react to one or more callbacks of a training loop
    """

    def __init__(self):
        # Dictionary containing functions to be called on change of specific values in the CallbackHandler state dict
        self.val_callbacks = {}

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
