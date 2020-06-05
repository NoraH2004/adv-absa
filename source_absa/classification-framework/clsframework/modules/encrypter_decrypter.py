import torch
from security import Encryption

from clsframework import Callback


class Decrypter(Callback):
    """Class to decrypt models on model load
    """

    def __init__(self):
        # Dictionary containing functions to be called on change of specific values in the CallbackHandler state dict
        self.val_callbacks = {}

    def on_model_load(self, model, **kwargs):
        if hasattr(model.config, 'do_encrypted') and model.config.do_encrypted:
            model_state_dict = model.state_dict()
            # Unshuffle the first layer in the state dict
            layer_to_shuffle = list(model_state_dict)[0]
            # Get classifier weights as a numpy array
            permuted_layer_np = model_state_dict[layer_to_shuffle].cpu().numpy()
            # Reverse permutation of classifier weights
            unpermuted_layer = Encryption.un_p(permuted_layer_np)
            # Convert permuted weights back to Torch tensor
            model_state_dict[layer_to_shuffle] = torch.from_numpy(unpermuted_layer)

            # Update model config
            model.load_state_dict(model_state_dict)
            return {'model': model}

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


class Encrypter(Callback):
    """Class to encrypt models on model save
    """

    def __init__(self, encrypt):
        self.encrypt = encrypt  # Boolean for whether to encrypt the model or not
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

    def on_model_save(self, model, **kwargs):
        if (self.encrypt):
            # Update configuration file
            model.config.do_encrypted = True

            # Get layer to shuffle as a numpy array
            model_state_dict = model.state_dict()
            layer_to_shuffle = list(model_state_dict)[0]  # Shuffle the first layer in the state dict
            layer_np = model_state_dict[layer_to_shuffle].cpu().numpy()

            # Permute classifier weights
            permuted_layer = Encryption.p(layer_np)

            # Convert permuted weights back to Torch tensor
            model_state_dict[layer_to_shuffle] = torch.from_numpy(permuted_layer)

            # Load updated state dict
            model.load_state_dict(model_state_dict)
            return {'model': model}
        else:
            # Update configuration file
            model.config.do_encrypted = False
            return {'model': model}
