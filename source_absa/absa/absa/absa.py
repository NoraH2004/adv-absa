from __future__ import absolute_import
from absa.utils import get_model_dir, get_aspects, predict_aspect_sentiments, get_text_aspect_pairs, get_config, Metrics
from clsframework.classifier import Classifier
from clsframework.callbackhandler import CallbackHandler
from clsframework.modules import ShuffleData, Train, ProgressReporter, Decrypter, ScoreReporter, Encrypter, \
    LearningRateReporterAndChanger, TorchOptimizer, AutoBatchsizeSelector, EarlyStopping, BestModelKeeper, \
    CustomTrainingStopper
from torch_optimizer import Ranger
import torch
import os
from security.authorization import Authorization
import logging
from clsframework.utils import fix_seed

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger_level = os.getenv('DO_LOGGING', 'DEBUG')
logger.setLevel(logger_level)
DEFAULT_SENTIMENT_LABELS = ['NONE', 'NEG', 'NEU', 'POS']


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class Predictor:
    """ A predictor class to generate ABSA predictions

    Parameters:
    model_folder (str): A path or name of the model that should be used
    """

    def __init__(self, model_folder, token=None, state_callback=None, stop_callback=None):
        logger.debug('Initializing Predictor')
        model_folder = get_model_dir(model_folder, token=token)
        self.device = get_device()
        self.aspects = get_aspects(model_folder=model_folder)
        self.config = get_config(os.path.join(model_folder, 'config.json'))
        callbacks = [Decrypter()]
        if state_callback is not None:
            callbacks.append(ProgressReporter(reporter=state_callback))
        if stop_callback is not None:
            callbacks.append(CustomTrainingStopper(stop_training_callback=stop_callback))
        ch = CallbackHandler(callbacks)
        self.absa = Classifier(model=model_folder, device=self.device, ch=ch)

    def predict(self, documents, token, aspects=None, with_segments=False):
        """Predicts given documents and returns a list of aspect sentiments

        Parameters:
        documents (list): A list of texts
        documents (list): A list of aspects
        token (string): A JSON web token which is validated

        Returns:
        aspect_sentiments (list of list): For each document a list of aspect_sentiments
       """
        # Check Authorization
        authorization_response = Authorization.authorize(token, functionality='Analysis/Aspect-Sentiments')
        if not authorization_response['verified']:
            logger.warning('User token not authorized to make predictions')
            raise ValueError(authorization_response['message'])
        # Predict Response
        if aspects is None:
            aspects = self.aspects
        aspect_sentiments = predict_aspect_sentiments(model=self.absa, aspects=aspects, documents=documents,
                                                      config=self.config, with_segments=with_segments)
        return aspect_sentiments


def evaluate(model_folder, documents, token):
    """ Evaluates a trained model on a dataset/annotated documents.

    Example:
        The annotated documents can be in 2 different formats exemplified here.

        >>> docs_with_segments = [{'segments': [{'text': 'Text', 'aspects':[['Room','POS']]}]}]
        >>> docs_simplified    = [{'aspect_sentiments':[{'aspect': 'Room', 'sentiment': 'POS',
        ... 'text': 'The room was very clean'}]}]

    Args:
        model_folder(str): Path to folder containing the ML model.
        annotated_documents(list): Annotated documents, 2 different formats supported.
        token(str): Model authorization token.

    Returns:
        dict: ABSA performance metrics are returned for the aspects, sentiments and combined.

    """
    # Authorize and Validate
    authorization_response = Authorization.authorize(token=token, functionality='absa/evaluate')
    if not authorization_response['verified']:
        raise ValueError(authorization_response['message'])

    # load model and aspects
    model_folder = get_model_dir(model_folder, token=token)
    aspects = get_aspects(model_folder=model_folder)

    # load model
    ch = CallbackHandler([Decrypter()])
    absa = Classifier(model=model_folder, device=get_device(), ch=ch)

    # predict dataset
    X, Y_true = get_text_aspect_pairs(documents, DEFAULT_SENTIMENT_LABELS, none_label_weight=1)
    Y_pred = absa(X=X)

    # compute performance metrics
    metrics = Metrics().calc_score_absa(X, Y_true, Y_pred, DEFAULT_SENTIMENT_LABELS, aspects)
    return metrics


def train(model_folder, documents, aspects, target, token,
          batchsize=32, encrypt=True, state_callback=None,
          stop_callback=None, epochs=20, validation_documents=None, seed=None,
          none_label_weight=0.5, learning_rate=2e-5, save_interval=None):
    """Trains a model with given annotated documents.

    Args:
        model_folder (str): A path or name of the base model that should be used.
        documents (list): An annotated list of documents.
        validation_documents (list): An annotated list of documents.
        target: (string): A path where the model should be saved.
        token (string): A JSON web token which is validated.
        batchsize (int): The batchsize used for training.
        epochs (int): The maximum number of epochs for training.
        encrypt (boolean): A flag whether to encrypt the model or not, default *True*.
        save_interval (int): Number of global steps after which a model checkpoint is saved.
        state_callback: Callback that is executed regularly and gives progress and score params back.
   """
    # Authorize and Check
    authorization_response = Authorization.authorize(token, functionality='absa/train')
    if not authorization_response['verified']:
        raise ValueError(authorization_response['message'])
    model_folder = get_model_dir(model_folder, token=token)
    # Preconditions
    if seed is not None:
        fix_seed(seed)
    # Load and Train Model
    handlers = [Decrypter(), ShuffleData(),
                Train(), TorchOptimizer(optclass=Ranger, lr=learning_rate),
                LearningRateReporterAndChanger(),
                Encrypter(encrypt)]
    if batchsize is None:
        handlers.append(AutoBatchsizeSelector(basemodel='bert-base-uncased', maxbatchsize=batchsize, verbose=True))
    if stop_callback is not None:
        handlers.append(CustomTrainingStopper(stop_training_callback=stop_callback))
    if state_callback is not None:
        handlers.append(ProgressReporter(reporter=state_callback))
    X, Y = get_text_aspect_pairs(documents=documents, labels=DEFAULT_SENTIMENT_LABELS,
                                 none_label_weight=none_label_weight)
    dataset_size = len(X)
    if validation_documents is not None and state_callback is not None:
        X_test, Y_test = get_text_aspect_pairs(documents=validation_documents, labels=DEFAULT_SENTIMENT_LABELS)

        # Custom evaluator for classification framework
        def evaluator(Y_true, Y_pred):
            return Metrics().calc_score_absa(X=X_test, Y_true=Y_true, Y_pred=Y_pred, labels=DEFAULT_SENTIMENT_LABELS,
                                             aspects=aspects
                                             )

        handlers.append(ScoreReporter(reporter=state_callback, evaluator=evaluator, X=X_test, Y=Y_test))
        BestModelKeeper(metricname='val_loss', maximize=False, verbose=True)
        handlers.append(
            EarlyStopping(patience=dataset_size * 2, lr_reduction_patience=dataset_size, lr_reduction_factor=0.1,
                          metricname='val_loss', verbose=True, epsilon=0.0001, maximize=False))

    ch = CallbackHandler(handlers)
    absa = Classifier(model=model_folder, num_labels=len(DEFAULT_SENTIMENT_LABELS), device=get_device(), ch=ch)
    absa.model.config.__dict__['do_aspects'] = aspects
    absa.model.config.__dict__['id2label'] = {id: label for id, label in enumerate(DEFAULT_SENTIMENT_LABELS)}
    absa.model.config.__dict__['label2id'] = {label: id for id, label in enumerate(DEFAULT_SENTIMENT_LABELS)}
    absa.model.train()
    absa(X, Y, batchsize=batchsize, verbose=True, epochs=epochs)
    # Save Model and Aspects
    absa.save(target)
    # Save to config


def suggest(model_folder, documents, token, size=200, epochs=2):
    """Creates text aspect suggestions that the model is unsure about

    Parameters:
    model_folder (str): A path or name of the base model that should be used
    documents (list): list of documents
    token (string): a JSON web token which is validated
    size (int): The amount of suggestions which should be returned
    epochs (int): The maximum number of epochs for training

    Returns:
    suggestions (list): a list of suggestions [{ 'text': 'This is a text', 'aspect': 'Drinks' }]
   """
    # Authorize and Check
    authorization_response = Authorization.authorize(token, functionality='absa/suggest')
    if not authorization_response['verified']:
        raise ValueError(authorization_response['message'])

    return []
