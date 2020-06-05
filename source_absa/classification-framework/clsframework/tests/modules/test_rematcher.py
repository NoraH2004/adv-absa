import os
import torch
from torch import nn
from clsframework import Classifier
from clsframework.modules import RematcherTrain, RematcherInference
from clsframework.modules.rematcher import RematcherModel, rematcher_optimizer


def test_rematchermodel():
    # Test of the RematcherModel and its capacity to be optimized.
    # First Step: Fake data 120 samples with 4 categories
    prediction = torch.tensor([0]*30+[1]*30+[2]*30+[3]*30, dtype=int)
    index = range(120)
    # We choose the following initial probabilities before rematch (clsprob):
    # For the the predicted categorie (see variable prediction above) the
    # probabilites are 70% while for the 3 others categories the probabilites
    # are at 10% each. We achieve this in two steps:
    clsprob = torch.zeros([120, 4]) + 0.1
    clsprob[index, prediction[index]] += 0.6   # 0.6 + 0.1 -> 70%
    # Now for the true labels:
    # We choose 9 of ten labels to match the predictions above but one of
    # ten labels wrong. With 4 categories, there are 3 wrong answers.
    # We take each wrong answer once. Hence, for each of the 4 categories
    # we need 30 predictions-labels-pairs where in 27 cases the lable equals
    # the prediction and in 3 cases lable <> prediction
    label = torch.tensor([0]*27 + [1, 2, 3] + [1]*27 + [2, 3, 0]+[2]*27 +
                         [3, 0, 1] + [3]*27+[0, 1, 2], dtype=int)
    # We have the fake data. Now we feed it into the model
    clsproblog = clsprob.log()
    rematcher_model = RematcherModel(4)
    criterion = nn.NLLLoss()
    start_loss = criterion(clsproblog, label)
    rematcher_model.backup_best(start_loss)
    rematcher_optimizer(model=rematcher_model, input=clsproblog, label=label,
                        make_log=False)
    rematcher_model.restore_best()
    assert 0.4349 < rematcher_model.best_loss < 0.4350
    # Moreover, all diagonal elements of rematcher_model.linear.weight
    # should be 1.520 and the offdiagonal elements -0.1735.
    # The bias should be close to zero.
    # print(rematcher_model.linear.weight)
    # But checking the loss should suffice under normal conditions


def test_rematch_moduls():
    # RematchTrain and RematchInference are tested together since RematchTrain
    # writes parameters to the disk which are needed for RematchInference.
    # Test for RematchTrain
    test_filename = "./Rematch_Test.pt"
    cls = Classifier(model="albert-base-v2", num_labels=4, device="cpu")
    X = [("Hello World", "Earth"), ("The world is nice", "Cleanliness")]
    Y = [1, 1]
    rematch_train = RematcherTrain(X, Y, test_filename, verbose=False)
    train_dict = rematch_train.on_inference_end(classifier=cls)
    # Task is so simple, that the loss should reach zero
    assert train_dict["loss after rematch"] < 1e-6  # we can expect much better

    # Test for RematchInference
    with torch.no_grad():
        results = [cls.forward(X, batchsize=2)]
    rematch_infer = RematcherInference(test_filename)
    infer_dict = rematch_infer.on_batch_end(results)
    # Results should match with Y (above) within numerical boundaries
    assert 0.999 < infer_dict["results"][0][0, 1] < 1.001
    assert 0.999 < infer_dict["results"][0][1, 1] < 1.001

    # Clean up
    os.remove(test_filename)
