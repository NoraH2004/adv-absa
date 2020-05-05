# Adversarial Examples in NLP using BERT

## BERT base Sentiment Classification

Sentiment Classification with bert-base-multilingual-uncased model finetuned for sentiment analysis on product reviews in six languages.

It predicts the sentiment of the review as a number of stars (between 1 and 5).

https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment

| Language | Accuracy (exact) | Accuracy (off-by-1) |
| -------- | ---------------------- | ------------------- |
| English  | 67%                 | 95%
| German   | 61%                 | 94%


## Adversarial Examples 

**Adversarial examples** are small, and often imperceptible perturbations applied to the input data in an effort to fool a deep classifier to incorrect classification.
These examples are a way to highlight model vulnerabilities and are useful for evaluation and interpretation of machine learning models. 

I generate adversarial text to attack a BERT based model used for Sentiment Classification by conducting non-targeted attacks in the white-box setting.

Using the **leave-one-out method** for the 'important word' detection, I determine the word which has a critical influence on the model's prediction.
I removing each word of a sentence one by one and let the model predict the incomplete sentences. 
Comparing the prediction before and after a word is removed reflects how the word influences the classification result. This procedure allows me to enhance the efficiency of my attacks.

To execute the perturbations, I focus on the input level rather than the embedding or semantic level. 

## Perturbations

I execute the attacks using three methods:

##### **133t 5p34k**
Testing the effect of Leet Speak on the BERT model
##### **Mispeelings**
Using the [Birkbeck file](https://www.dcs.bbk.ac.uk/~ROGER/corpora.html "List of common typos"), a list of common typos documented by the Oxford Text Archive. The file contains 36,133 misspellings of 6,136 words.
##### **,Punctuation.?**
Testing the influence of additional or missing punctuation marks.


The result is an adversarial dataset. 

## Results

|  | Leet Speak | Typos | Punctuation |
| -------- | ---------------------- | ------------------- | ------------------- |
|Tokenizer / Model | nlptown/bert-base-multilingual-uncased-sentiment | nlptown/bert-base-multilingual-uncased-sentiment | nlptown/bert-base-multilingual-uncased-sentiment
|Dataset | TripAdvisor Hotel Reviews | TripAdvisor Hotel Reviews | TripAdvisor Hotel Reviews
|Size original Dataset | 435 | 435 | 435
|Size adversarial Dataset | 183 | 330 | 56
|Modification Ratio | 42.01% | 75.86% | 12.87%
