import json 

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



