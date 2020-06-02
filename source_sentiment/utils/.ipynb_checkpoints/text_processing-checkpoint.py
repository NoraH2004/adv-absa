import json 
import pandas as pd

def load_jsonline(filename, limit):
    data = []
    with open(filename) as f:
        counter = 0
        for line in f:
            counter += 1
            py_obj = json.loads(line)
            data.append(py_obj)
            if counter > limit:
                break
    return data


# tokenization

def detokenize(tok_sentence):
    sentence = ' '.join(tok_sentence)
    return sentence

def get_token_dropped_sentence_at_pos(sent,token):
    tok_mod_sentence = sent.copy()    
    tok_mod_sentence.pop(token)
    return tok_mod_sentence


# Functions to generate perturbations

def to_leet(word):
    getchar = lambda c: chars[c] if c in chars else c
    chars = {"a":"4","e":"3","l":"1","o":"0","s":"5"}
    return ''.join(getchar(c) for c in word)

def to_typo(typodict, word):
    if word in typodict:
        return typodict[word]
    else: return None

def to_punctuation(word):
    
    return ''.join((word, ','))


def generate_modified_sentences(original_sentences, important_words, modified_words):
    assert len(original_sentences)==len(important_words)==len(modified_words), 'List length is not equal!'
    
    modified_sentences = []
    for index, sentence in enumerate(original_sentences):
        if modified_words[index] is None:
            modified_sentences.append(sentence)
            continue  
            
        if isinstance(modified_words[index], list): 
            modified_sentences_list = []
            for word in modified_words[index]:
                modified_sentences_list.append(sentence.replace(important_words[index], word))
            modified_sentences.append(modified_sentences_list)               
            continue        
        modified_sentences.append(sentence.replace(important_words[index], modified_words[index]))   
        
    return modified_sentences

def predict_sentiment(model, tokenizer, sentence):
    inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
    prediction = model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'])[0].argmax().item()
    
    return prediction



def filter_unchanged_predictions(ds):
    
#    filters the dataframe
#    remove all rows that have an unchanged prediction
#    if row includes a list of sentences, remove all unchanged predictions from list


    # filter lists
    for index, item in ds.iterrows():
        if not isinstance(item.modified_sentence, list):
            continue

        if len(item.modified_prediction) != len(item.modified_sentence):
            raise Exception('could not filter df row [' + str(index) + '], length of modified_prediction and modified_sentence does not match')

        filtered_sentences = []
        filtered_predictions = []

        for i, sententence in enumerate(item.modified_sentence):
            prediction = item.modified_prediction[i]
            if item.original_prediction != prediction:
                filtered_sentences.append(sententence)
                filtered_predictions.append(prediction)

        # optional, if list has one element, unpack
        if len(filtered_predictions) == 1:
            filtered_sentences = filtered_sentences[0]
            filtered_predictions = filtered_predictions[0]
        
        ds.at[index,'modified_sentence'] = filtered_sentences
        ds.at[index,'modified_prediction'] = filtered_predictions

    # remove empty lists
    ds = ds[ds.modified_sentence.str.len() != 0] 

    # filter non-lists
    ds = ds[ds.original_prediction != ds.modified_prediction]

    return ds



# Funcitons for Result Vizualisation


def generate_results_df(pmethod, ds, advds, *args):        
        
    results = pd.DataFrame({
     'Perturbation Method': [pmethod],
     'Tokenizer': ['nlptown/bert-base-multilingual-uncased-sentiment'], 
     'Model' : ['nlptown/bert-base-multilingual-uncased-sentiment'], 
     'Dataset':['TripAdvisor Hotel Reviews'], 
     'Output lables': ['Range from 0 to 4 - 0 = NEG; 4 = POS'],
     'Items in original dataset': len(ds),
     'Number of modifyable items original': len(advds),
     'Items in adversarial dataset': len(advds),
     'Percentage': (len(advds)/len(ds)*100)})
    
    return results.T


def generate_multipredictions(original_predictions, modified_predictions):
    extended_original_predictions = []
    extended_modified_predictions = []

    for index, prediction in enumerate(original_predictions):

        if isinstance(modified_predictions[index], list):
            for e, item in enumerate(modified_predictions[index]):            
                extended_original_predictions.append(original_predictions[index])
                extended_modified_predictions.append(modified_predictions[index][e])
        else:
            extended_original_predictions.append(original_predictions[index])
            extended_modified_predictions.append(modified_predictions[index])

    return extended_original_predictions, extended_modified_predictions
