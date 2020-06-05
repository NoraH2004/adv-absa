from clsframework import Classifier, CallbackHandler
from clsframework.modules import ModelSampler
from clsframework.active_learning import select_segments


def test_active_learning():
    saps = [("What a nice hotel", "Hotel"),
            ("The room was horrible", "Staff"),
            ("What's up with the bathroom?", "Room")]*2
    cls = Classifier(model="bert-base-uncased", num_labels=4, device="cpu", ch=CallbackHandler([ModelSampler()]))
    logits_B_K_C = []
    for i in range(10):
        logits_B_K_C.append(cls.forward(saps, verbose=True))
    idxs, score = select_segments(logits_B_K_C, acquisitionsize=4, preselectionsize=15, samples=100, verbose=True)
    print("score=", score)
    for i in idxs:
        print(saps[i])
