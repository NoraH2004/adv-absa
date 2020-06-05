import os
from functools import lru_cache

import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)

from clsframework import (BatchIterator, CallbackHandler,
                          CUDAOutOfMemoryException, StopTrainingException)


class CachedAutoTokenizer():
    """A version of AutoTokenizer where the tokenization of single sentences is cached.

    This greatly increases speed when the same sentence is tokenized
    multiple times as is the case for the BERT pair architecture
    """
    def __init__(self, model):
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    @lru_cache(maxsize=2**20)
    def _get_input_ids(self, text):
        assert isinstance(text, str)
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def encode_plus(self, text, text_pair=None, **kwargs):
        if isinstance(text, tuple):
            text_pair = text[1]
            text = text[0]
        first_ids = self._get_input_ids(text)
        second_ids = self._get_input_ids(text_pair) if text_pair is not None else None
        return self.tokenizer.prepare_for_model(first_ids, second_ids, max_length=512, **kwargs)

    @lru_cache(maxsize=2**20)
    def tokenlength(self, *args, **kwargs):
        """Returns how many tokens a given text or text_pair would result in. Results are cached for performance reasons."""
        return len(self.encode_plus(*args, **kwargs)["input_ids"])


class Classifier():
    """A Class implementing a general text, text_pair classifier
    """
    def __init__(self, model, device="cuda", num_labels=None, fp16=None, ch=CallbackHandler([])):
        self.ch = ch  # Callback Handler
        self.nbiterations = 0   # How many iterations have been performed
        self.nbseensamples = 0  # How many samples have been seen by the system
        # Check if cuda is available and set device to cpu if it is not
        if not torch.cuda.is_available() and device != "cpu":
            print(f"device {device} was requested, but CUDA not found. Reverting to CPU")
        # If we are using cpu as a device, fp16 is not available
        if device == "cpu" and fp16 is not None:
            print(f"CPU was chosed as device and FP16 opt level of {fp16} was chosen. FP16 is only supported on CUDA")
            fp16 = None
        self.fp16 = fp16  # Remember what fp16 mode was requested
        self.set_fp16 = None  # Remember what fp16 mode the model was set to

        # Load config so we can change the number of labels to use if needed
        self.config = AutoConfig.from_pretrained(model)
        if num_labels is not None:
            self.config.num_labels = num_labels
        # Load model from pretrained file using config
        self.model = AutoModelForSequenceClassification.from_pretrained(model, config=self.config)

        # ############# On Model Load #############
        self.model, device, self.fp16 = self.ch.on_model_load(model=self.model, device=device, fp16=fp16)
        # ##############################################

        # Move model to requested device
        self.device = device
        self.model.eval()
        self.model.to(self.device)
        # Load tokenizer
        self.tokenizer = CachedAutoTokenizer(model)

    @staticmethod
    def _stack_tokens(tokendicts):
        # Determine maximum number of tokens
        maxtokens = max(len(val) for t in tokendicts for val in t.values())
        # Create empty numpy arrays that can hold all tokens
        resultdict = {k: np.zeros((len(tokendicts), maxtokens)) for k in tokendicts[0].keys()}
        # Stack the different outputs into numpy-arrays
        for i in range(len(tokendicts)):
            for key, value in tokendicts[i].items():
                resultdict[key][i, 0:len(value)] = value
        # If token_type_ids is all zero, remove from dict since
        # RoBERTa has problems if it is sent through the model and all
        # other models will be unaffected because it is 0 anyway
        if "token_type_ids" in resultdict.keys() and np.sum(resultdict["token_type_ids"]) == 0.0:
            del resultdict["token_type_ids"]

        return resultdict

    def _batch_forward(self, x, y=None):
        """Forward pass through the model. Returns class probabilities of
        given sentences/sentence pairs (given as strings or already as
        a tokenized dict). If labels are also given, returns loss.

        samples can be either a list of strings for single sentence
        prediction or a list of tuples of strings for prediction using
        two sentences (e.g. sentence aspect pairs).
        """
        if y is not None:
            assert len(x) == len(y)
        # Tokenize if we are given raw strings
        if type(x[0]) != dict:
            tokendicts = [self.tokenizer.encode_plus(sample) for sample in x]
        # Stack individual token lists into numpy-arrays
        tokens = self._stack_tokens(tokendicts)
        # Convert to tensors and push to used device
        tokens = {key: torch.tensor(value, dtype=torch.long).to(self.device) for key, value in tokens.items()}
        if y is not None:
            y = torch.tensor(y, dtype=torch.long).to(self.device)
        # Perform forward pass and return result (don't calc gradient if we only evaluate)
        try:
            if y is not None:  # We are training
                result = self.model(labels=y, **tokens)[0]
                self.nbiterations += 1
                self.nbseensamples += len(x)
            else:
                with torch.no_grad():
                    result = self.model(labels=y, **tokens)[0]

            # ############# After Forward Pass #############
            x, y, result, self.nbiterations, self.nbseensamples = self.ch.after_forward_pass(
                x=x, y=y, result=result, nbiterations=self.nbiterations,
                nbseensamples=self.nbseensamples)
            # ##############################################
        except RuntimeError as e:
            # Check if we actually have an out of memory error since there is no real exception for it
            if 'out of memory' in str(e):
                raise CUDAOutOfMemoryException(str(e))
            else:
                raise e
        return result

    def forward(self, X, Y=None, batchsize=8, epochs=1, verbose=False):  # noqa: C901 # ignore complexity warning
        """Forward pass through the model. Returns class probabilities of
        given sentences/sentence pairs (given as strings or already as
        a tokenized dict). If labels are also given, returns loss.

        samples: can be either a list of strings for single sentence
        prediction or a list of tuples of strings for prediction using
        two sentences (e.g. sentence aspect pairs).

        batchsize: Maximal number of samples that are sent through the
        model at once

        batchcallback: A function that is called on the result of the
        forward pass after each batch

        verbose: If true, a progress bar is shown (default False)

        shuffle: If true, the order in which the samples are sent
        through the model are randomly shuffled. The labels are
        shuffled in the same way.
        """

        # ############# Inference Begin #############
        X, Y, batchsize, self.model, self.tokenizer, epochs, _ = self.ch.on_inference_begin(
            X=X, Y=Y, batchsize=batchsize, model=self.model, tokenizer=self.tokenizer,
            epochs=epochs, classifier=self)
        # ###########################################

        # Set model to fp16 if requested and model has not yet been
        # converted (e.g. by an optimizer)
        if self.fp16 is not None and self.set_fp16 is None:
            try:
                import apex
                self.model, _ = apex.amp.initialize(self.model, [], opt_level=self.fp16)
                self.set_fp16 = self.fp16
            except ModuleNotFoundError:
                print("Not able to set model to FP16 since apex is not installed")
                self.fp16 = None

        epochsit = tqdm(range(epochs), desc="Epoch") if verbose else range(epochs)
        try:
            for epoch in epochsit:
                # ############# Epoch Begin #############
                X, Y, batchsize, self.model, self.tokenizer, _ = self.ch.on_epoch_begin(
                    X=X, Y=Y, batchsize=batchsize, model=self.model, tokenizer=self.tokenizer, epoch=epoch)
                # #######################################

                results = []

                # Create batchiterator
                batches = BatchIterator(X, Y, batchsize=batchsize, verbose=verbose)

                # Run forward for all batches and call batchcallback after each batch
                for x, y in batches:
                    # ############# Batch Begin #############
                    x, y = self.ch.on_batch_begin(x=x, y=y)
                    # #######################################

                    # Perform forward pass and execute callback
                    res = self._batch_forward(x, y)
                    # Aggregate results
                    if Y is None:
                        results.append(res.cpu().detach().softmax(1).numpy())
                    else:
                        results.append(res.item())
                    # ############## Batch End ##############
                    results = self.ch.on_batch_end(results=results)
                    # #######################################

                # Stack results into one numpy array
                results = np.vstack(results)

                # ############## Epoch End ##############
                results = self.ch.on_epoch_end(results=results)
                # #######################################
        except StopTrainingException:
            results = np.vstack(results)

        # ############## Inference End ##############
        self.model = self.ch.on_inference_end(model=self.model)
        # ###########################################

        # Output results depending on the wanted format (loss when
        # Y are given, class probabilities if no Y are
        # given)
        if Y is None:
            return results
        else:
            # Weight the loss by the number of X in a batch for
            # more consistent results while shuffeling and return mean
            # loss per sample
            results = [l * batches.batchsizes[i] for i, l in enumerate(results[:, 0])]
            result = np.sum(results) / len(X)
            return result.item()

    def __call__(self, X, Y=None, **kwargs):
        """Forward pass through the model. Returns predicted class of
        given sentences/sentence pairs (given as strings or already as
        a tokenized dict). If labels are also given, returns loss.

        Rest of the parameters see method "forward(...)"
        """
        result = self.forward(X, Y, **kwargs)
        if Y is None:
            return np.argmax(result, axis=1)
        else:
            return result

    def save(self, directory):
        """Saves model as well as tokenizer info to directory
        """
        # Create dir if it does not exists
        if not os.path.isdir(directory):
            os.mkdir(directory)
            # Save model and vocabulary

        # ############## On Model Save ##############
        self.model, self.tokenizer = self.ch.on_model_save(model=self.model, tokenizer=self.tokenizer)
        # ###########################################

        self.model.save_pretrained(directory)
        self.tokenizer.tokenizer.save_pretrained(directory)
