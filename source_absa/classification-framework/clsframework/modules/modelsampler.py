from collections.abc import Iterable
from clsframework import Callback, CallbackHandler
import torch


class MonteCarloDropout(torch.nn.Module):
    """Class replacing original pytorch Dropout where the dropout can be
    fixed and also be used during inference to simulate drawing
    samples from a probability distribution over trained models.
    """
    __constants__ = ['p', 'inplace']

    def __init__(self, p=0.5, inplace=False, maxbatchsize=128, maxtokens=512):
        super().__init__()

        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        self.active = False     # If False dropout is not applied
        self.p = p              # Dropout probability
        self.inplace = inplace
        self.maxbatchsize = maxbatchsize  # Biggest expected batch size
        self.maxtokens = maxtokens        # Highest expected number of tokens
        # The rest of the arguments are dictionaries since we have to
        # consider shared dropout layers which might have to operate
        # on input of different dimensionality (e.g. ALBERT is an
        # example for such an architecture)
        self.mask = {}          # Big dropout mask we can cut a piece out of to suite our current input size
        self.cachedmask = {}    # Current cutout that will be reused if inputsize stays the same
        self.last_input_shape = {}  # Saves last input shapes for all dimensionalities
        self.batchsizedims = {}     # Dimensions corresponding to the batchsize in input
        self.tokenlengthdims = {}   # Dimensions corresponding to the tokenlendths in input

    def _get_max_mask_shape(self, shape):
        """Returns the maximal expected size for the mask we will need. Takes
        into consideration which dimensions encode for the batchsize
        and the number of tokens and selects the highest expected
        value for these
        """
        shapedims = len(shape)
        resshape = list(shape)

        for i in self.batchsizedims[shapedims]:
            # Our code only works if batchsize is encoded along the first dimension
            assert i == 0, f"Only works in cases where the batchsize is encoded along the first dimension but got {i}"
            resshape[i] = self.maxbatchsize
        for i in self.tokenlengthdims[shapedims]:
            resshape[i] = self.maxtokens

        return tuple(resshape)

    def clear_mask(self):
        """Clear the current mask so a new one is randomly generated on the next forward pass"""
        self.mask = {}
        self.cachedmask = {}
        self.cachedmaskshape = {}

    def _create_mask(self, input: torch.Tensor):
        """Create a new mask, fitting for the size of the input tensor"""
        ishlen = len(input.shape)
        # Determine maximal size of mask we will ever need
        maxmask = self._get_max_mask_shape(input.shape)
        # Sample random mask for one sample of the batch
        self.mask[ishlen] = torch.empty(maxmask[1:], dtype=torch.uint8, device=input.device).bernoulli_(self.p)
        self.mask[ishlen] = self.mask[ishlen].bool()
        # Expand mask along the number of samples from one batch (maximal expected batches)
        self.mask[ishlen] = self.mask[ishlen].unsqueeze(0).expand(maxmask[0], *([-1] * (input.dim() - 1)))

    def apply_dropout(self, input: torch.Tensor):
        """Returns the input with the currently defined mask applied"""
        ishlen = len(input.shape)
        # Remember input shape to be able to determine max possible shape
        self.last_input_shape[ishlen] = input.shape

        if self.active:
            # If we don't have a mask, generate one
            if ishlen not in self.mask:
                self._create_mask(input)
            # Cut out mask if necessary
            if ishlen in self.cachedmask and input.shape == self.cachedmask[ishlen].shape:
                cutmask = self.cachedmask
            else:
                cutmask = self.mask[ishlen][tuple(slice(0, e) for e in input.shape)]
                self.cachedmask[ishlen] = cutmask
            # Apply dropout and scale result (so that average sum for a
            # neuron stays the same during training and inference)
            return input.masked_fill(cutmask, 0) / (1 - self.p)
        else:
            return input

    def forward(self, input: torch.Tensor):
        """Function called on forward pass through model"""
        # If we are in training mode reset mask in every inference
        # step. Normally we would not use our own Dropout for
        # training, but to be safe ...
        if self.training:
            self.clear_mask()
            # Apply dropout
        return self.apply_dropout(input)


def monte_carlo_dropout_patch(model, seen=None, **kwargs):
    """Takes a transformer model, tries to find all dropout layers
    in it and replaces them with a MonteCarloDropout with given keyword arguments"""
    if seen is None:
        seen = []
    dropoutlayers = []
    # Go through all atributes of object
    for d in dir(model):
        # Get objects associated to the attribute
        att = getattr(model, d)
        # Ensure that we don't get infinite recursion by ignoring objects we have already seen
        if id(att) in seen:
            continue
        seen.append(id(att))
        # If we find a dropout layer, replace it with our own
        if "Dropout" in str(type(att)):
            setattr(model, d, MonteCarloDropout(att.p, **kwargs))
            dropoutlayers.append(getattr(model, d))
        # If we find something iterable, recursevly call function for all elements
        elif isinstance(att, Iterable):
            for i in att:
                dropoutlayers = dropoutlayers + monte_carlo_dropout_patch(i, seen, **kwargs)
        # For all other objects that are part of transformers, call function recursevly
        elif "transformers" in str(type(att)):
            dropoutlayers = dropoutlayers + monte_carlo_dropout_patch(att, seen, **kwargs)
    return dropoutlayers


class ModelSampler(Callback):
    """Exchanges Dropout of the current model to one also active during
    inference. Fixes the dropout for once epoch during inference.
    Mainly used to get samples for active learning.
    """
    def __init__(self):
        super().__init__()
        self.dropoutlayers = None
        # Register value update callbacks
        # self.val_callbacks["test"] = self.test_change_callback

    def on_model_load(self, model, **kwargs):
        self.dropoutlayers = monte_carlo_dropout_patch(model)
        # Set all dropout layers to eval mode
        for i in self.dropoutlayers:
            i.eval()
        return {"model": model}

    def on_inference_begin(self, classifier, tokenizer, **kwargs):
        """Determines required mask sizes for all the Dropout layers"""
        tmpch = classifier.ch
        classifier.ch = CallbackHandler([])
        # Some models need to be in train mode to work correctly?
        classifier.model.train()
        # Try to determine which dimensions of the input to the
        # dropout layers could vary with tokenlength and batchsize
        probestring = "a a a a a a a a a a as sd we sd"
        tokenlength = classifier.tokenizer.tokenlength(probestring)
        batchsize = 7
        classifier([probestring]*batchsize)
        for dl in self.dropoutlayers:
            for dimsize, last_input_shape in dl.last_input_shape.items():
                dl.tokenlengthdims[dimsize] = [i for i, v in enumerate(last_input_shape) if v == tokenlength]
                dl.batchsizedims[dimsize] = [i for i, v in enumerate(last_input_shape) if v == batchsize]
        # A second pass with a different number of tokens and a
        # different batchsize might be needed here to check the
        # dimensions if we ever get problems ...

        # Activate Dropout layers
        for dl in self.dropoutlayers:
            dl.active = True
        # Restore original callbackhandler
        classifier.ch = tmpch

    def on_epoch_begin(self, **kwargs):
        # Clear all dropout masks so new ones will be created for the next epoch
        for d in self.dropoutlayers:
            d.clear_mask()

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
