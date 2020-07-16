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

def to_punctuation(word):
    
    return ''.join((word, ','))


def generate_modified_sentence_packages(original_sentences_unfiltered, important_words_packages, modified_words_packages):
    assert len(original_sentences_unfiltered)==len(important_words_packages)==len(modified_words_packages), 'List length is not equal!'
    
    len_mod_sent = 0
    modifyable_original_sentences = []
    modified_sentence_packages = []
    modified_sentence_list_lvl0 = []
    for i, sentence in enumerate(original_sentences_unfiltered):
        iw_per_sentence = modified_words_packages[i]
       # print('iw_per_sentence: ', iw_per_sentence)
        

        modified_sentences_word_list = []        
        for e, word_variances in enumerate(iw_per_sentence):
            iw_variances_per_word = iw_per_sentence[e]
            # print('iw_variances_per_word: ', iw_variances_per_word)
            
            modified_sentences_word_variances = []
            for word in iw_variances_per_word:
                #print('word: ', word) 
                if isinstance(word, list):
                    for variance in word:
                        # print('variance: ', variance)
                        if important_words_packages[i][e] != variance and variance is not None:
                            mod_sent = sentence.replace(important_words_packages[i][e],variance)
                            modified_sentences_word_variances.append(mod_sent) 
                            len_mod_sent += 1

                else:
                    # print(word)
                    if important_words_packages[i][e] != word and word is not None:
                        mod_sent = sentence.replace(important_words_packages[i][e],word)
                        modified_sentences_word_variances.append(mod_sent) 
                        len_mod_sent += 1
                                  
                    
            if modified_sentences_word_variances:
                modified_sentences_word_list.append(modified_sentences_word_variances)
            
          
        if modified_sentences_word_list:
            modified_sentence_packages.append(modified_sentences_word_list)
            modifyable_original_sentences.append(sentence)
            
                    
    return modifyable_original_sentences, modified_sentence_packages, len_mod_sent
    


def predict_sentiment(model, tokenizer, sentence):
    inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
    prediction = model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'])[0].argmax().item()
    
    return prediction



# generate result dict

def compare_results(original_predictions, modified_predictions):
    
    #function comapres original with modified predictions and returns dictionary, with only those, where the modification changed the prediction
    
    d_results = []

    for e, modified_predictions_set in enumerate(modified_predictions):

        original_prediction = original_predictions[e]
        modified_sentences = []
        modified_aspect_sentiments = []

        for modified_sentence in modified_predictions_set:
            if modified_sentence['aspect_sentiments'] != original_prediction['aspect_sentiments']:
                modified_sentences.append(modified_sentence['text'])
                modified_aspect_sentiments.append(modified_sentence['aspect_sentiments'])

        if modified_sentences:
            d_results.append(
                    {
                        'original_sentence': original_prediction['text'],
                        'original_result': original_prediction['aspect_sentiments'],
                        'modified_sentences': modified_sentences,
                        'modified_results': modified_aspect_sentiments
                    })

    return d_results



def generate_results_lists(d_results):
    
    # function returns single lists of results, suitable for pandas df
    
    original_texts = []
    original_results = []
    modified_texts = []
    modified_results = []
    
    successfull_modifications = 0

    for item in d_results:
        original_texts.append(item['original_sentence'])
        original_results.append(item['original_result'])

        modified_text_p_sent = []
        modified_result_p_sent = []
        for sentence in item['modified_sentences']:
            modified_text_p_sent.append(sentence)
            successfull_modifications += 1
        for result in item['modified_results']:
            modified_result_p_sent.append(result)
        modified_texts.append(modified_text_p_sent)
        modified_results.append(modified_result_p_sent)

    return original_texts, original_results, modified_texts, modified_results, successfull_modifications



  



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


def generate_results_df(original_sentences, modifyable_original_sentences, number_of_modified_sentences, successfull_modifications, pmethod):        
    try:
        success_rate = successfull_modifications/number_of_modified_sentences
    except ZeroDivisionError:
        success_rate = 0
    
    results = pd.DataFrame({
     'Perturbation Method': [pmethod],
     'Tokenizer': ['en_core_web_sm'], 
     'Model' : ['en-laptops-absa'], 
     'Dataset':['SemEval 2015 Laptops'], 
     'Total number of original sentences': len(original_sentences),
     'Total number of modifyable original sentences': len(modifyable_original_sentences),
     'Total number of modified sentences': number_of_modified_sentences,
     'Total number of changed predictions through modification': successfull_modifications,
     'Success Rate': success_rate})
    
    return results.T

def generate_multipredictions(original_results, modified_results):
    extended_original_results = []
    extended_modified_results = []

    for index, result in enumerate(original_results):
        for e, item in enumerate(modified_results[index]):            
            extended_original_results.append(original_results[index])
            extended_modified_results.append(modified_results[index][e])

    return (extended_original_results, extended_modified_results)


def generate_aspsent_map_dict(results_dict):

    # check, which aspects and sentiments are in the data

    aspects = []
    sentiments = []
    aspects_sentiments = []
    aspects_sentiments_map = []
    i = -1

    for item in results_dict:
        #print()
        for result in item['original_result']:
            if result['aspect'] not in aspects:
                aspects.append(result['aspect'])
            if result['sentiment'] not in sentiments:
                sentiments.append(result['sentiment'])
            aspect_sentiment = []
            aspect_sentiment.append(result['aspect'])
            aspect_sentiment.append(result['sentiment'])
            if aspect_sentiment not in aspects_sentiments:
                aspects_sentiments.append(aspect_sentiment)
                i += 1
                #aspects_sentiments_map.append(i)

                aspects_sentiments_map.append(
                    {i:aspect_sentiment})

    return aspects_sentiments_map
