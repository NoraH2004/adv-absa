import pytest
from clsframework import StopTrainingException
from clsframework.modules import EarlyStopping


def test_simulated_stopping():
    stop = EarlyStopping(patience=5, metricname="val_accuracy", maximize=True, verbose=True)
    stop.on_metric_change(nbseensamples=1, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=2, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=3, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=4, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=5, val_accuracy=0.1)
    with pytest.raises(StopTrainingException):
        stop.on_metric_change(nbseensamples=6, val_accuracy=0.1)

    stop = EarlyStopping(patience=3, metricname="val_accuracy", maximize=True, verbose=True)
    stop.on_metric_change(nbseensamples=1, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=2, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=3, val_accuracy=0.15)
    stop.on_metric_change(nbseensamples=4, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=5, val_accuracy=0.1)
    with pytest.raises(StopTrainingException):
        stop.on_metric_change(nbseensamples=6, val_accuracy=0.1)

    stop = EarlyStopping(patience=3, metricname="val_accuracy", maximize=False, verbose=True)
    stop.on_metric_change(nbseensamples=1, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=2, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=3, val_accuracy=0.05)
    stop.on_metric_change(nbseensamples=4, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=5, val_accuracy=0.1)
    with pytest.raises(StopTrainingException):
        stop.on_metric_change(nbseensamples=6, val_accuracy=0.1)

    stop = EarlyStopping(patience=3, metricname="val_accuracy", maximize=True, verbose=True)
    stop.on_metric_change(nbseensamples=1, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=2, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=3, val_accuracy=0.15)
    stop.on_metric_change(nbseensamples=4, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=5, val_accuracy=0.1)
    with pytest.raises(StopTrainingException):
        stop.on_metric_change(nbseensamples=6, val_accuracy=0.1)

    stop = EarlyStopping(patience=3, metricname="val_accuracy", maximize=False, verbose=True)
    stop.on_metric_change(nbseensamples=1, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=2, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=3, val_accuracy=0.05)
    stop.on_metric_change(nbseensamples=4, val_accuracy=0.1)
    stop.on_metric_change(nbseensamples=5, val_accuracy=0.1)
    with pytest.raises(StopTrainingException):
        stop.on_metric_change(nbseensamples=6, val_accuracy=0.1)


def test_simulated_lr_adaptation():
    stop = EarlyStopping(patience=50, metricname="val_accuracy", maximize=True, lr_reduction_patience=2, verbose=True)
    lr = 1.0
    stop.on_metric_change(nbseensamples=1, val_accuracy=0.1, lr=lr)
    stop.on_metric_change(nbseensamples=1, val_accuracy=0.1, lr=lr)
    stop.on_metric_change(nbseensamples=2, val_accuracy=0.1, lr=lr)
    lr = stop.on_metric_change(nbseensamples=3, val_accuracy=0.1, lr=lr)["lr"]
    stop.on_metric_change(nbseensamples=4, val_accuracy=0.1, lr=lr)
    lr = stop.on_metric_change(nbseensamples=5, val_accuracy=0.1, lr=lr)["lr"]
    stop.on_metric_change(nbseensamples=6, val_accuracy=0.1, lr=lr)
    lr = stop.on_metric_change(nbseensamples=7, val_accuracy=0.1, lr=lr)["lr"]
    assert lr == 1/8
