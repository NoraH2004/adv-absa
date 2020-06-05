from clsframework import CallbackHandler, Classifier
from clsframework.modules import Evaluator


def test_eval():
    # Prepare "validation dataset"
    valX = ["Hello", "Bla", "Blu"]
    valY = [1, 0, 1]
    # Set up model
    ch = CallbackHandler([Evaluator(X=valX, Y=valY, interval=1, verbose=True)])
    cls = Classifier("albert-base-v2", num_labels=4, device="cpu", ch=ch)
    # Execute Evaluator
    for i in range(20):
        _ = ch.on_batch_end(nbseensamples=i+1, classifier=cls)
        assert "val_accuracy" in ch.state.keys()
