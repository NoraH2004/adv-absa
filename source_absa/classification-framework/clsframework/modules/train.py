import torch

from clsframework import Callback


class Train(Callback):
    """Implements backprop and gradient clipping as a module
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

    def on_batch_begin(self, model, **kwargs):
        model.train()

    def after_forward_pass(self, y, result, model, classifier, optimizer=None, **kwargs):
        if y is not None:
            # Calculate gradients using backprop
            if classifier.set_fp16 is None:
                result.backward()
                # Clip gradients that are too big
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            elif optimizer is not None:
                from apex import amp
                with amp.scale_loss(result, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # Clip gradients that are too big
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm=1.0)
            else:
                print("Tried to use apex amp during training, but no optimizer is registered!")
        else:
            raise Exception("Tried to run backprop, but no training labels were given")

    def on_batch_end(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_inference_end(self, **kwargs):
        pass

    def on_model_save(self, **kwargs):
        pass
