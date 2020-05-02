
def generate_modified_sentences(original_sentences, important_words, modified_words):
    assert len(original_sentences)==len(important_words)==len(modified_words), 'this is an error'
    
    modified_sentences = []
    for index, sentence in enumerate(original_sentences):
        if modified_words[index] is not None:
            modified_sentences.append(sentence.replace(important_words[index], modified_words[index]))
        else: modified_sentences.append(sentence)
    return modified_sentences



# Functions to generate perturbations

def to_leet(word):
    getchar = lambda c: chars[c] if c in chars else c
    chars = {"a":"4","e":"3","l":"1","o":"0","s":"5"}
    return ''.join(getchar(c) for c in word)

def to_typo(typodict, word):
    if word in typodict:
        return typodict[word]
    else: return None

def to_punct(word):
    return ''.join((word, ','))


def predict_sentiment(model, tokenizer, sentence):
    inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
    prediction = model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'])[0].argmax().item()
    return prediction


