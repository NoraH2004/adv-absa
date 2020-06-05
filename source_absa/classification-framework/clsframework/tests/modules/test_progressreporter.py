from clsframework import CallbackHandler
from clsframework.modules import ProgressReporter


def test_progessreporter():

    reported_progress = 0

    def reporter(progress):
        nonlocal reported_progress
        reported_progress = progress

    progress_reporter = ProgressReporter(reporter=reporter)
    ch = CallbackHandler([progress_reporter])
    valX = ["Hello", "This", "is", "a", "Test"]

    # 5 samples x 10 epochs = 50 all_samples
    ch.on_inference_begin(X=valX, epochs=10)
    assert 50 == progress_reporter.all_samples

    # Simulate training
    for i in range(50):
        ch.after_forward_pass(x=["Hello"])
        assert reported_progress == (i+1)*2

    # If all samples are seen, 100% must be reached
    assert reported_progress == 100
