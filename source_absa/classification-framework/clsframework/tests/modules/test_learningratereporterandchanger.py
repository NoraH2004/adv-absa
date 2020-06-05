from clsframework import Classifier, CallbackHandler
from clsframework.modules import TorchOptimizer, LearningRateReporterAndChanger, EarlyStopping, Evaluator
from torch.optim import Adam


def test_lrchange():
    X = ["zero", "one", "two", "three"]
    Y = [0, 1, 2, 3]

    ch = CallbackHandler([
        TorchOptimizer(optclass=Adam, lr=0.01),
        Evaluator(X=X, Y=Y, interval=1),
        EarlyStopping(patience=10, lr_reduction_patience=5),
        LearningRateReporterAndChanger(),
    ])
    classifier = Classifier(model="albert-base-v2", num_labels=4, device="cpu", ch=ch)
    classifier(X, Y)
    # Check of lr is the default of 0.01
    assert ch["lr"] == 0.01
    classifier(X, Y, epochs=100)
    # Check if early stopping has worked
    assert ch["nbseensamples"] == 16
    # Check if lr has decreased after multiple steps of non-increasing accuracy
    assert ch["lr"] < 0.01
    # Check if lr was written back to optimizer
    assert ch["optimizer"].param_groups[0]["lr"] == ch["lr"]
