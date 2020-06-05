from clsframework import Callback


class ModuleTemplate(Callback):
    """
    """
    def __init__(self):
        super().__init__()
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
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_inference_end(self, **kwargs):
        pass

    def on_model_save(self, **kwargs):
        pass
