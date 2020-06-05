import sys

import torch
import torch.cuda as tc
from torch.utils.tensorboard import SummaryWriter

from clsframework import Callback


class TensorBoardLogger(Callback):
    """Implements TensorBoard logging. Also provides the created
    summarywriter to be used by other modules if needed
    """

    def __init__(self, comment=""):
        super().__init__()
        self.comment = comment

    def on_model_load(self, **kwargs):
        pass

    def on_inference_begin(self, summarywriter=None, **kwargs):
        if summarywriter is None:
            self.sw = SummaryWriter(comment="_" + self.comment)
            self.sw.add_text("Base/Arguments", " ".join(sys.argv), 0)
            return {"summarywriter": self.sw}

    def on_epoch_begin(self, **kwargs):
        pass

    def on_batch_begin(self, x, nbseensamples=0, **kwargs):
        self.sw.add_scalar("Base/Batchsize", len(x), nbseensamples)

    def after_forward_pass(self, y, result, nbseensamples, lr=None, **kwargs):
        # Only log loss if we have labels to calculate loss with
        if y is not None:
            self.sw.add_scalar("Base/Trainloss", result.item(), nbseensamples)

        if torch.cuda.is_available():
            # Log CUDA memory information if available
            self.sw.add_scalar("CUDA/Mem_allocated", tc.memory_allocated(), nbseensamples)
            # self.sw.add_scalar("CUDA/Max Mem allocated", tc.max_memory_allocated(), nbseensamples)
            self.sw.add_scalar("CUDA/Mem_reserved", tc.memory_reserved(), nbseensamples)
            # self.sw.add_scalar("CUDA/Max Mem reserved", tc.max_memory_reserved(), nbseensamples)
            self.sw.add_scalar("CUDA/Mem_cached", tc.memory_reserved(), nbseensamples)
            # self.sw.add_scalar("CUDA/Max Mem cached", tc.max_memory_reserved(), nbseensamples)

        # Log learning rate if applicable
        if lr is not None:
            self.sw.add_scalar("Base/Learningrate", lr, nbseensamples)

        # Flush so information is written immediatly
        self.sw.flush()

    def on_batch_end(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_inference_end(self, **kwargs):
        pass

    def on_model_save(self, **kwargs):
        pass
