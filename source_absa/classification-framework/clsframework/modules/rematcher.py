import torch
from torch import nn
from torch.optim import LBFGS
from clsframework import Callback, CallbackHandler
from clsframework.modules import SortByLength


def rematcher_optimizer(model, input, label, make_log=False):
    # LBFGS is a second order optimizer; stong_wolfe makes it more stable
    optimizer = LBFGS(model.parameters(), line_search_fn="strong_wolfe",
                      history_size=59, max_iter=60)
    criterion = nn.NLLLoss()
    if make_log:
        input = input.log()

    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output.log(), label)
        model.backup_best(loss)
        loss.backward()
        return loss
    optimizer.step(closure)


class RematcherModel(nn.Module):
    """
    A RematcherModel is a small network that takes the output log-probabilities
    of another (deep) model like e.g. BERT and calculates new probabilies.
    This might be beneficial when the deep model was trained on an unbalanced
    training set. Since the RematcherModel is very small, it is unlikely to
    overfit and hence can be optimized on small data sets (The Rematchermodel
    contains only n * (n + 1) parameters, where n is the number of categories.)
    The RematcherModel might also be useful, when the deep model is not fully
    optimized (e.g. due to early stopping).
    The RematcherModel consists of a linear layer and a softmax, which are
    initalized such that they act as an identity (apart from mapping
    log-probabilities to probabilities). This garantees that the loss of
    the RematcherModel after optimization is always equal or better than the
    original loss of the deep model alone.
    """
    def __init__(self, number_of_categories):
        super().__init__()
        self.linear = nn.Linear(number_of_categories, number_of_categories,
                                bias=True)
        self.best_weight = torch.eye(number_of_categories)
        self.best_bias = torch.zeros(number_of_categories)
        # best_loss should be initialized by an external call of backup_best
        # Here, we take a huge number
        self.best_loss = 1e8
        self.restore_best()  # here it works as initializer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        shift = (x.max() + x.min()).detach() / 2
        # The scalar "shift" does not influence the results of the softmax
        # but makes it numerically more stable
        x = self.softmax(x-shift)
        return x

    def backup_best(self, last_loss):
        # This function is called after each optimization step to memorize the
        # best results in case the optimization goes wrong at some later step
        if last_loss < self.best_loss:
            with torch.no_grad():
                self.best_loss = last_loss.item()
                self.best_weight.copy_(self.linear.weight)
                self.best_bias.copy_(self.linear.bias)

    def restore_best(self):
        # The logical counterpart to "backup_best"
        with torch.no_grad():
            self.linear.weight.copy_(self.best_weight)
            self.linear.bias.copy_(self.best_bias)


class RematcherTrain(Callback):
    """
    When on_inference_end is called, a new RematcherModel is created, trained
    and its parameters are stored to the specified file.
    """
    def __init__(self, X, Y, filename="./rematch_param.pt", verbose=False):
        # X: sentence-pairs; Y: label;
        # filename: where the parameters of the RematcherModel are stored
        super().__init__()
        self.X = X
        self.Y = torch.tensor(Y)
        self.filename = filename
        self.verbose = verbose
        self.ch = CallbackHandler([SortByLength()])

    def on_model_load(self, **kwargs):
        pass

    def on_inference_begin(self, classifier, **kwargs):
        if not(classifier.device == "cuda" and classifier.fp16 is None):
            # unfortunatedly, we don't no why, yet
            raise Exception("Rematcher requires fp16 is None.")

    def on_epoch_begin(self, **kwargs):
        pass

    def on_batch_begin(self, **kwargs):
        pass

    def after_forward_pass(self, **kwargs):
        pass

    def on_batch_end(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_inference_end(self, classifier, **kwargs):
        # Switch CallbackHandler
        chtmp = classifier.ch
        classifier.ch = self.ch
        # Run inference on given data
        model_trains = classifier.model.training
        classifier.model.eval()
        with torch.no_grad():
            results = classifier.forward(self.X, batchsize=128,
                                         verbose=self.verbose)
        if model_trains:
            classifier.model.train()
        # Switch back to original callback handler
        classifier.ch = chtmp
        # Rematch
        clsprob = torch.tensor(results)
        clsproblog = clsprob.log()
        number_of_categories = clsprob.shape[1]
        rematcher_model = RematcherModel(number_of_categories)
        criterion = nn.NLLLoss()
        with torch.no_grad():
            start_loss = criterion(clsproblog, self.Y)
        rematcher_model.backup_best(start_loss)
        rematcher_optimizer(rematcher_model, clsproblog, self.Y, False)
        rematcher_model.restore_best()
        if self.verbose:
            print("Loss before rematch: ", start_loss.item())
            print("Loss after rematch: ", rematcher_model.best_loss)
            print("Weight and bias of rematch-model: ")
            print(rematcher_model.linear.weight.detach())
            print(rematcher_model.linear.bias.detach())
        torch.save(rematcher_model.state_dict(), self.filename)
        return {"loss before rematch": start_loss.item(),
                "loss after rematch": rematcher_model.best_loss}

    def on_model_save(self, **kwargs):
        pass


class RematcherInference(Callback):
    """
    When on_batch_end is called, a RematcherModel is instantiated and its
    parameters are set to the values found in 'filename'. Subsequent the
    RematcherModel is applied to the input 'results'.
    """
    def __init__(self, filename="./rematch_param.pt"):
        # filename: where the parameters of the RematcherModel are stored
        super().__init__()
        # Loading the parameters of the RematchModel
        try:
            statedict = (torch.load(filename))
        except Exception as e:
            error_msg = "RematcherInference was not able to load the " \
                      + "trained weights file: " + str(e)
            raise IOError(error_msg)
        number_of_categories = statedict["linear.bias"].shape[0]
        self.rematch_model = RematcherModel(number_of_categories)
        self.rematch_model.load_state_dict(statedict)
        self.rematch_model.eval()

    def on_model_load(self, **kwargs):
        pass

    def on_inference_begin(self, classifier, **kwargs):
        if not(classifier.device == "cuda" and classifier.fp16 is None):
            # unfortunatedly, we don't no why, yet
            raise Exception("Rematcher requires fp16 is None.")

    def on_epoch_begin(self, **kwargs):
        pass

    def on_batch_begin(self, **kwargs):
        pass

    def after_forward_pass(self, **kwargs):
        pass

    def on_batch_end(self, results, **kwargs):
        # Uses the last probabilities in the results-list and takes
        # the logarithm which is needed for the RematchModel
        with torch.no_grad():
            result_log = torch.tensor(results[-1]).log()
        # Do the calculation
        with torch.no_grad():
            rematched_results = self.rematch_model(result_log)
        # Replacing the probabilities of the main model with
        # the new rematched probabilities
        results[-1] = rematched_results.detach().cpu().numpy()
        return {"results": results}

    def on_epoch_end(self, **kwargs):
        pass

    def on_inference_end(self, **kwargs):
        pass

    def on_model_save(self, **kwargs):
        pass
