from __future__ import absolute_import, division, print_function, unicode_literals
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, mean_absolute_error
import logging
import json
import requests
from zipfile import ZipFile
import random
import numpy
from nlp import LanguageDetection, SentenceSegmentation
from collections import Counter

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger_level = os.getenv('DO_LOGGING', 'DEBUG')
logger.setLevel(logger_level)


# GENERAL UTILS #

# Load aspects
def get_aspects(config_dict=None, model_folder=None, aspects_path=None):
    # Try to load aspects from the config by default
    if config_dict is not None:
        if 'do_aspects' in config_dict:
            aspects = config_dict['do_aspects']
            logger.debug('Loaded aspects from config')
            return aspects

    # Next try to get aspects from the config in model_folder
    if model_folder is not None:
        config_dict = get_config(os.path.join(model_folder, 'config.json'))
        if 'do_aspects' in config_dict:
            aspects = config_dict['do_aspects']
            logger.debug('Loaded aspects from config')
            return aspects
        else:
            aspects_path = os.path.join(model_folder, 'aspects.jsonl')

    # Finally, try to load aspects from a file path
    if aspects_path is not None:
        with open(aspects_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
        aspects = [json.loads(line)['name'] for line in lines]
        logger.debug(f"Aspects loaded from {aspects_path}")
        return aspects

    raise ValueError(
        'Aspects not found. Please specify a file path to an aspects file, or a folder containing a config containing do_aspects.')


# Get config dictionary object of model
def get_config(config_path):
    with open(config_path) as file:
        config = json.load(file)
    logger.debug(f"Config loaded from {config_path}")
    return config


def get_model_dir(model_folder, token=None):
    current_file_dir = os.path.dirname(__file__)
    logger.debug(f"Loading model {model_folder}")
    # If model_folder is a file path
    if model_folder is not None and os.path.exists(model_folder):
        return model_folder
    # If model_folder is the name of a model in 'models' folder
    elif os.path.exists(os.path.join(current_file_dir, 'models', model_folder)):
        return os.path.join(current_file_dir, 'models', model_folder)
    # If model is not found, download the model
    elif token is not None and model_folder in requests.get('https://download.deepopinion.ai/models').json():
        # Create models folder if not yet given
        if not os.path.exists(os.path.join(current_file_dir, 'models')):
            os.makedirs(os.path.join(current_file_dir, 'models'))
        # Prepare
        model_url = "https://download.deepopinion.ai/models/" + model_folder
        logger.debug(f"Attempting to download model from {model_url}")
        zip_path = os.path.join(current_file_dir, 'models', model_folder + '.zip')
        # Download the model
        response = requests.get(
            model_url,
            allow_redirects=True,
            headers={"authorization": "Bearer " + token}
        )
        # Write the downloaded .zip file to local storage
        if response.status_code == 200:
            open(zip_path, 'wb').write(response.content)
            # Open the .zip file in read mode and extract all files
            with ZipFile(zip_path, 'r') as zip:
                logger.debug(f"Extracting file {zip_path}")
                zip.extractall(path=os.path.join(current_file_dir, 'models'))
            # Remove the zip file
            os.remove(zip_path)
            # Return the path to the newly downloaded model
            return os.path.join(current_file_dir, 'models', model_folder)
        # If model can't be successfully downloaded, raise an error
        else:
            logger.error(
                f"Model could not be downloaded. Server returned response code {response.status_code}.")
            raise ValueError('Model could not be downloaded')
    else:
        logger.debug("Using the model from transformers storage!")
        return model_folder


def get_text_aspect_pairs(documents, labels, none_label_weight=1):
    X = []
    Y = []
    for doc in documents:
        if 'aspect_sentiments' in doc:
            for a_s in doc['aspect_sentiments']:
                if a_s['sentiment'] == 'NONE' and random.random() <= none_label_weight:
                    continue
                X.append((a_s['text'], a_s['aspect']))
                Y.append(labels.index(a_s['sentiment']))
        elif 'segments' in doc:
            for s in doc['segments']:
                for a in s['aspects']:
                    if a[1] == 'NONE' and random.random() <= none_label_weight:
                        continue
                    X.append((s['text'], a[0]))
                    Y.append(labels.index(a[1]))

    assert (len(X) == len(Y))
    return X, Y


# PREDICTION UTILS #

def predict_aspect_sentiments(model, aspects, documents, config, with_segments=False):
    id2label = {int(k): v for k, v in config['id2label'].items()}
    none_index = next(key for key, value in id2label.items() if value == 'NONE')
    ld = LanguageDetection()
    segmenter = SentenceSegmentation()
    # Split documents
    text_aspect_pairs = []
    all_segments = []
    result_helper = []
    segment_index = 0
    for doc_index, doc in enumerate(documents):
        doc = doc if isinstance(doc, dict) else {'text': doc}
        language = doc['language'] if 'language' in doc else ld.detect(doc['text'])
        segments = doc['segments'] if 'segments' in doc else segmenter.split(document=doc['text'], langcode=language)[
            'segments']
        for segment in segments:
            all_segments.append(segment)
            for aspect in aspects:
                text_aspect_pairs.append((segment['text'], aspect))
                result_helper.append((doc_index, segment_index))
            segment_index += 1
    # Inference / actual prediction
    y_result = model(X=text_aspect_pairs)
    # Shape to human readable format
    result = [[] for _ in documents]
    for i, y in enumerate(y_result):
        doc_index = result_helper[i][0]
        segment_index = result_helper[i][1]
        if not with_segments and y != none_index:
            result[doc_index].append(
                {'aspect': text_aspect_pairs[i][1], 'sentiment': id2label[y],
                 'text': text_aspect_pairs[i][0], 'span': all_segments[segment_index]['span']}
            )
        if with_segments:
            segments = [segment for segment in result[doc_index]]
            found_segment = next((s for s in segments if s['text'] == text_aspect_pairs[i][0]), None)
            if found_segment and y != none_index:
                found_segment['aspect_sentiments'].append({'aspect': text_aspect_pairs[i][1], 'sentiment': id2label[y]})
            elif not found_segment:
                segment = {'text': text_aspect_pairs[i][0],
                           'span': all_segments[segment_index]['span'],
                           'aspect_sentiments': []
                           }
                if y != none_index:
                    segment['aspect_sentiments'].append({'aspect': text_aspect_pairs[i][1], 'sentiment': id2label[y]})
                result[doc_index].append(segment)
    return result


# EVAL UTILS #

class Metrics(object):
    """ Calculates metrics from prediction and ground truth data.

    Contains functionality to compute performance scores for ABSA and other tasks.

    """

    def _input_to_multiaspect_format(self, X, Y, labels):
        """ Aggregates segment_aspect_pairs into a multilabel format assuming ordered segment-aspect pairs

        Examples:
            The inputs have the following format:
            >>> X = [('Segment 1', 'Aspect 1'), ('Segment 1', 'Aspect 2'), ('Segment 2', 'Aspect 3')]
            >>> Y_true = [2,1,3]
            >>> Y_pred = [0,0,3]
        Args:
            X(list(tuple)): segment-aspect pairs
            Y(list): Sentiment labels of segment-aspect pairs
            labels: All sentiment classes

        Returns:
            list(list)

        """
        multiaspectsentiment_per_segment = []
        last_segment = "###some random unique string###"
        for (segment, aspect), sentiment in zip(X, Y):
            if segment != last_segment:
                # strings only
                multiaspectsentiment_per_segment.append([[aspect, labels[sentiment]]])
            else:
                # append to the last item of the array
                multiaspectsentiment_per_segment[-1].append([aspect, labels[sentiment]])

        return multiaspectsentiment_per_segment

    def _convert_to_multilabel_multioutput(self, multi_aspect_sentiments, aspects, labels):
        """ Converts multi_aspect_sentiments format into 2 dimensional multilabel-multioutput

        Args:
            multi_aspect_sentiments: list([["Room","NONE"], ...])
            aspects(list): All aspect categories.
            labels(list): All sentiment labels including *NONE*

        Returns:
            numpy.array(N,M): N examples, M total aspects, array entries are sentiment_idx

        """
        multilabel_multioutput = []

        for aspect_sentiments in multi_aspect_sentiments:
            row = [0] * len(aspects)
            for aspect_sentiment in aspect_sentiments:
                aspect_name = aspect_sentiment[0]
                sentiment_name = aspect_sentiment[1]
                aspect_idx = aspects.index(aspect_name)
                sentiment_idx = labels.index(sentiment_name)
                row[aspect_idx] = sentiment_idx
            multilabel_multioutput.append(row)

        return numpy.array(multilabel_multioutput)

    def _multilabel_multioutput_to_aspect_format(self, multilabel_multioutput):
        """ Converts multilabel-multioutput into aspect one-hot multilabel format

        Examples:
            The output is a matrix which has the following format: One-hot multilabel
            4 Aspects - 2 segments total (aggregated):
            >>> X = [('Segment 1', 'Aspect 1'), ('Segment 1', 'Aspect 2'), ('Segment 2', 'Aspect 3')]
            >>> Y_true = numpy.array([[1,1,0,0],[0,0,1,0]])
            >>> Y_pred = numpy.array([[0,0,0,0],[0,0,1,0]])
        Returns:
            numpy.array(N,M)
        """
        aspect_format = numpy.copy(multilabel_multioutput)
        aspect_format[aspect_format != 0] = 1

        return aspect_format

    def _multilabel_multioutputs_to_sentiment_format(self, ml_multioutput_true, ml_multioutput_pred):

        y_true = numpy.hstack(ml_multioutput_true)
        y_pred = numpy.hstack(ml_multioutput_pred)
        y_true_sentiment = [
            y_true[i] for i in range(len(y_true)) if y_pred[i] != 0 and y_true[i] != 0
        ]
        y_pred_sentiment = [
            y_pred[i] for i in range(len(y_true)) if y_pred[i] != 0 and y_true[i] != 0
        ]

        return y_true_sentiment, y_pred_sentiment

    def _compute_aspect_metrics(self, y_true, y_pred):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_micro': precision_score(y_true, y_pred, average='micro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_micro': recall_score(y_true, y_pred, average='micro'),
            'f1_weighted': fbeta_score(y_true, y_pred, 1, average='weighted'),
            'f1_macro': fbeta_score(y_true, y_pred, 1, average='macro'),
            'f1_micro': fbeta_score(y_true, y_pred, 1, average='micro'),
        }
        return metrics

    def _compute_detailed_aspect_metrics(self, y_true, y_pred, ml_aspectsentiments_true, allaspects):
        details = {}
        aspects_list = []
        for aspectsentiments in ml_aspectsentiments_true:
            for aspect_sentiment in aspectsentiments:
                aspect, sentiment = aspect_sentiment[0], aspect_sentiment[1]
                if sentiment != "NONE":
                    aspects_list.append(aspect)
        aspect_counter = Counter(aspects_list)
        for i in range(y_true.shape[1]):
            details[allaspects[i]] = self._roundobj({'f1': round(fbeta_score(y_true[:, i], y_pred[:, i], 1), 2),
                                                     'precision': round(precision_score(y_true[:, i], y_pred[:, i]), 2),
                                                     'recall': round(recall_score(y_true[:, i], y_pred[:, i]), 2),
                                                     'support': aspect_counter[allaspects[i]]
                                                     })
        return details

    def _compute_sentiment_metrics(self, y_true, y_pred):
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                '1-MAE': 1 - mean_absolute_error(y_true, y_pred),
                'f1_weighted': fbeta_score(y_true, y_pred, 1, average='weighted'),
                'f1_macro': fbeta_score(y_true, y_pred, 1, average='macro'),
                'f1_micro': fbeta_score(y_true, y_pred, 1, average='micro'),
            }
        except ValueError:
            logger.warning("No aspects matched to compute sentiment metrics!")
            metrics = {
                'accuracy': numpy.nan,
                '1-MAE': numpy.nan,
                'f1_weighted': numpy.nan,
                'f1_macro': numpy.nan,
                'f1_micro': numpy.nan,
            }
        return metrics

    def _roundobj(self, obj, len=2):
        if type(obj) == float or type(obj) == numpy.float64:
            obj = round(obj, len)
        elif type(obj) == list:
            for i in obj:
                i = round(i, len)
        elif type(obj) == dict:
            for k, v in obj.items():
                obj[k] = round(v, len)
        else:
            raise TypeError("Type not supported!")
        return obj

    def calc_score_absa(self, X, Y_true, Y_pred, labels, aspects):
        """ Computes performance scores for the ABSA task.

        For aspects the most important score is the F1-score whereas for sentiment
        1-MAE (Mean Absolute Error) is the most important score. For clients most often
        accuracy is most understandable.

        Examples:
            Input formats exemplified.

            >>> X = [('Segment 1', 'Room'), ('Segment 1', 'Cleanliness'), ('Segment 2', 'Staff')]
            >>> Y_true = [0,1,3] # = NONE, POS, NEG
            >>> Y_pred = [0,1,2] # = NONE, POS, NEU

            Most reportable score for customers.

            >>> scores = calc_score_absa(example_input, ...)
            >>> print(scores['combined']['weighted_accuracy'])

        Args:
            X(list(tuple)): List of segment-aspect pairs.
            Y_true(list(int)): Ground truth of segment-aspect pair.
            Y_pred(list(int)): Prediction of segment-aspect pair.
            labels(list): All possible sentiment labels.
            aspects(list): All possible aspects labels the model is trained on.

        Returns:
            dict: Returns an object containing aspect, sentiment, combined and scores for each aspect in detail.

        """

        # input -> multilabel_aspectsentiment
        ml_aspectsentiment_true = self._input_to_multiaspect_format(X, Y_true, labels)
        ml_aspectsentiment_pred = self._input_to_multiaspect_format(X, Y_pred, labels)

        # multilabel_aspectsentiment -> multilabel_multioutput
        ml_multioutput_true = self._convert_to_multilabel_multioutput(ml_aspectsentiment_true, aspects, labels)
        ml_multioutput_pred = self._convert_to_multilabel_multioutput(ml_aspectsentiment_pred, aspects, labels)

        # multilabel_multioutput -> aspect_format
        aspects_true = self._multilabel_multioutput_to_aspect_format(ml_multioutput_true)
        aspects_pred = self._multilabel_multioutput_to_aspect_format(ml_multioutput_pred)

        # multilabel_multioutput -> sentiment_format
        sentiments_true, sentiments_pred = self._multilabel_multioutputs_to_sentiment_format(ml_multioutput_true,
                                                                                             ml_multioutput_pred)
        aspect_metrics = self._compute_aspect_metrics(aspects_true, aspects_pred)

        detailed_aspect_metrics = self._compute_detailed_aspect_metrics(aspects_true,
                                                                        aspects_pred,
                                                                        ml_aspectsentiment_true,
                                                                        aspects)
        sentiment_metrics = self._compute_sentiment_metrics(sentiments_true, sentiments_pred)

        score = {
            'aspect': self._roundobj(aspect_metrics),
            'aspect_details': detailed_aspect_metrics,
            'sentiment': self._roundobj(sentiment_metrics),
            'combined': {
                'product': self._roundobj(aspect_metrics["f1_weighted"] * sentiment_metrics["1-MAE"]),
                'weighted': self._roundobj(0.3 * aspect_metrics["f1_weighted"] + 0.7 * sentiment_metrics["1-MAE"]),
                'weighted_accuracy': self._roundobj(
                    0.5 * aspect_metrics["accuracy"] + 0.5 * sentiment_metrics["accuracy"])
            }
        }

        return score
