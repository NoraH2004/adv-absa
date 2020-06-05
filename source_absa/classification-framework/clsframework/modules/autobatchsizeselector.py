import json
import os
import warnings

import pynvml

from clsframework import Callback


class AutoBatchsizeSelector(Callback):
    """Automatically determines batchsizes depending on the GPU and
    whether fp16 is being used or not. The selected batchsizes are
    intended for training, since for inference they don't really
    matter anyway.
    """
    def __init__(self, basemodel, maxbatchsize, verbose=False):
        super().__init__()
        self.basemodel = basemodel
        self.maxbatchsize = maxbatchsize
        self.verbose = verbose

        # Load limits file
        filename = os.path.join(os.path.dirname(__file__), "autobatchsizeselector.json")
        with open(filename) as io:
            self.limits = json.load(io)
        self.limits = self.limits
        # Register value update callbacks
        # self.val_callbacks["test"] = self.test_change_callback

    def _select_correct_limits(self, fp16):
        """Returns the limits (batchsize given a max tokenlength) for the
        given basemodel, fp16 and amount of free memory. If no valid
        limits could be found, None is returned
        """
        if self.basemodel in self.limits.keys() and fp16 in self.limits[self.basemodel].keys():
            limits = self.limits[self.basemodel][fp16]
        else:
            warnings.warn(f"AutoBatchsizeSelector: Basemodel [{self.basemodel}] and/or " +
                          f"FP16 mode [{fp16}] could not be found in database")
            return None

        # Look for the biggest free memory value that was tested that
        # is smaller than our actually free memory
        validmem = 0
        for mem in sorted([int(mem) for mem in limits.keys()]):
            if mem <= self.freemem:
                validmem = mem
        # If we do not find any valid memory size, be save and fall back to batchsize of 1
        if validmem == 0:
            warnings.warn("AutoBatchsizeSelector: No valid memory range could be determined")
            return None
        else:
            limits = limits[str(validmem)]
        if self.verbose:
            print(f"AutoBatchsizeSelector: Determined valid memory: {validmem}")

        return limits

    def _determine_max_batchsize(self, fp16, max_token_length):
        """Determine the max valid batchsize given a certain basemodel,
        fp16-mode and max token length
        """
        limits = self._select_correct_limits(fp16)
        if limits is None:      # If no valid limits were found, we revert to batchsize of 1
            warnings.warn("No valid limits found. Reverting to batchsize of 1")
            return 1
        # Look for biggest batch size we can use
        maxvalidbatchsize = 0
        tokenlengths = sorted([int(tokenlength) for tokenlength in limits.keys()])
        # Consider case where the smallest recorded tokenlength is still bigger than our max token length
        if tokenlengths[0] > max_token_length:
            maxvalidbatchsize = limits[str(tokenlengths[0])]
        for t in tokenlengths:
            if t <= max_token_length:
                maxvalidbatchsize = limits[str(t)]
        # If no valid batchsize has been found revert to batchsize of 1
        if maxvalidbatchsize == 0:
            warnings.warn("AutoBatchsizeSelector: No valid batchsize could be determined. Fallig back to batchsize of 1.")
            maxvalidbatchsize = 1
        if self.verbose:
            print(f"AutoBatchsizeSelector: Determined max batchsize: {maxvalidbatchsize}")

        return maxvalidbatchsize

    def on_model_load(self, **kwargs):
        pass

    # The amount of free GPU mem has to be determined once before any
    # training has occurred. After this point the tensors, gradients,
    # etc. already fill a large part of VRAM and it is hard to say how
    # much memory can actually be used
    def on_inference_begin(self, device, **kwargs):
        # Determine which GPU should be used. If multiple GPUs are
        # selected raise an exception since that is not supported
        # by this module
        if device == "cpu":
            warnings.warn("AutoBatchsizeSelector not supported/needed when using the CPU")
        else:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                if "," in os.environ["CUDA_VISIBLE_DEVICES"]:
                    raise Exception("Tried to use AutoBatchsizeSelector with multiple GPUs. This is not supported.")
                else:
                    devicenr = int(os.environ["CUDA_VISIBLE_DEVICES"])
            else:
                devicenr = 0    # If nothing specific is set use first GPU
            # Determine how much VRAM is free on device
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(devicenr)
            self.freemem = pynvml.nvmlDeviceGetMemoryInfo(h).free
            if self.verbose:
                print(f"AutoBatchsizeSelector_Free_VRAM: {self.freemem}")

    # We are setting the batchsize on_epoch_begin and not
    # on_inference_begin since the data could change for each epoch
    # (e.g. data augmentation etc.)
    def on_epoch_begin(self, X, classifier, device, **kwargs):
        if device == "cpu":
            warnings.warn("AutoBatchsizeSelector not supported/needed when using the CPU")
        else:
            # Determine max number of tokens of any sentence/sentence-pair
            max_token_length = max(len(classifier.tokenizer.encode_plus(x)["input_ids"]) for x in X)
            if self.verbose:
                print(f"AutoBatchsizeSelector_Max_token_length: {max_token_length}")
            # Select max possible batch size for given max number of tokens
            fp16_mode = "None" if classifier.set_fp16 is None else classifier.set_fp16
            max_batchsize = self._determine_max_batchsize(fp16=fp16_mode,
                                                          max_token_length=max_token_length)
            # Respect maximal requested batchsize
            batchsize = min(max_batchsize, self.maxbatchsize)
            if self.verbose:
                print(f"AutoBatchsizeSelector_Batchsize_to_be_used: {batchsize}")
            # Update batchsize that is used for this epoch
            return {"batchsize": batchsize}

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
