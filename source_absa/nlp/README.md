# NLP

Helper module for basic NLP tasks. 
Relies mainly on open source modules but serves as interface to guarantee stability/abstraction for DeepOpinion.

| Project Maintainer | Proxy|
| ------------- | ------------- |
| Paul Opitz | Alex Rietzler  |


![Build and Deploy NLP](https://github.com/deepopinion/nlp/workflows/Build%20and%20Deploy%20NLP/badge.svg?branch=master)


Currently available:
 * Sentence Segmentation
    * "EN" - English
    * "DE" - German
    * "FR" - French
 * Language Detection
 * Translation to:
    * "EN" - English
    * "DE" - German
    * "FR" - French
    * "ES" - Spanish
    * "PT" - Portuguese
    * "IT" - Italian
    * "NL" - Dutch
    * "PL" - Polish
    * "RU" - Russian
 
 ## Installation
 
 * Install and activate virtual environment

    ``python3.7 -m venv env``
    
    ``source env/bin/activate``
    
* Download spacy
    
    ``python -m spacy download de_core_news_sm``    
    ``python -m spacy download en_core_web_sm``    
    ``python -m spacy download fr_core_news_sm``
    
* Add DEEPL_KEY as environment variable if you want to use the translations feature

* Install module to run tests

    ``pip install -e .``    
    