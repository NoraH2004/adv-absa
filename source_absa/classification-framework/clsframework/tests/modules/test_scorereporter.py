from clsframework import CallbackHandler
from clsframework.modules import ScoreReporter
from clsframework.classifier import Classifier


def test_scorereporter():

    reported_score = None

    def reporter(score):
        nonlocal reported_score
        reported_score = score

    def evaluator(Y_true, Y_pred):
        accuracy = sum(Y_pred == Y_true) / len(Y_true)
        return {'combined': {'weighted_accuracy': accuracy}}

    X = [("Lasagna", "Food"), ("Aperol Spritz", "Drink")]
    Y = [0, 1]
    score_reporter = ScoreReporter(reporter=reporter, evaluator=evaluator, X=X, Y=Y)
    ch = CallbackHandler([score_reporter])
    cls = Classifier("albert-base-v2", num_labels=4, device="cpu", ch=ch)

    score_reporter.evaluate(classifier=cls)
    assert reported_score is not None

    # Simulate batches
    assert score_reporter.avg_batch_duration is None

    for i in range(10):
        ch.on_batch_begin()
        ch.on_batch_end(classifier=cls)
        if i < 3:
            assert len(score_reporter.batch_durations) == i+1
        else:
            assert len(score_reporter.batch_durations) == 3
        assert score_reporter.avg_batch_duration is not None

    assert reported_score is not None
