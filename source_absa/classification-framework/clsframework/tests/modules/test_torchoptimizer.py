from clsframework.modules import TorchOptimizer
from torch.optim import Adam


def test_instantiation():
    TorchOptimizer(optclass=Adam, grad_accumulation_steps=4)
