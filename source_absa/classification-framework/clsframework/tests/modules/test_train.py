from clsframework import CallbackHandler, Classifier
from clsframework.modules import Train


def test_eval():
    # Prepare a "dataset"
    X = ["Hello", "Bla", "Blu"]
    Y = [3, 3, 3]
    # Set up model
    ch = CallbackHandler([Train()])
    cls = Classifier("albert-base-v2", num_labels=4, device="cpu", ch=ch)
    # Execute "Training". Since we are not using an optimizer, no
    # weights will be updated. This only tests for runtime errors
    cls(X, Y)
