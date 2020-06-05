import os
import requests
import json


class Translation(object):

    def translate(self, documents, target='en', source=None, split=True):
        if len(documents) == 0:
            return "No documents given!"
        if len(documents) > 50:
            return "Too many documents given. Maximum 50 per call!"
        key = os.environ['DEEPL_KEY']
        data = {'auth_key': key, 'target_lang': target, 'text': documents}
        if source is not None:
            data['source_lang'] = source
        data['split_sentences'] = 1 if split else 0
        result = requests.post('https://api.deepl.com/v2/translate', data=data)
        if result.status_code == 200:
            translations = json.loads(result.content)['translations']
            return [t['text'] for t in translations]
        else:
            return result.status_code
