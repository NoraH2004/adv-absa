def semeval_to_aspectsentiment_hr(filename):
    sentimap = {
        'positive': 'POS',
        'negative': 'NEG',
        'neutral': 'NEU'
    }

    def transform_category_name(s):
        return s

    with open(filename) as file:

        review_elements = ET.parse(file).getroot().iter('Review')

        sentences = []
        aspect_category_sentiments = []
        classes = set([])
        sentence_ids = []

        for j, review_element in enumerate(review_elements):
            # review_text = ' '.join([el.text for el in review_element.iter('text')])

            for i, s in enumerate(review_element.iter('sentence')):
                s_id = s.get('id')
                sentence_ids.append(s_id)
                sentence_text = s.find('text').text
                aspect_category_sentiment = set([])
                for o in s.iter('Opinion'):
                    aspect_category = transform_category_name(o.get('category'))
                    classes.add(aspect_category)
                    sentiment = sentimap[o.get('polarity')]
                    aspect_category_sentiment.add((aspect_category, sentiment))

                # TODO how to deal with conflicting sentiments?!

                aspect_sentiment_dict = {}
                sentence_has_conflict = False
                for asentis in aspect_category_sentiment:
                    if asentis[0] in aspect_sentiment_dict:
                        print('Conflicting AspectSentiment detected: ', aspect_category_sentiment)
                        sentence_has_conflict = True
                    else:
                        aspect_sentiment_dict[asentis[0]] = asentis[1]
                if not sentence_has_conflict:
                    sentences.append(sentence_text)
                    aspect_category_sentiments.append(aspect_sentiment_dict)

        cats = list(classes)
        cats.sort()

    idx2aspectlabel = {k: v for k, v in enumerate(cats)}
    sentilabel2idx = {"NONE": 0, "NEG": 1, "NEU": 2, "POS": 3, "CONF": 4}
    idx2sentilabel = {k: v for v, k in sentilabel2idx.items()}
    if not with_ids:
        return sentences, aspect_category_sentiments, (idx2aspectlabel, idx2sentilabel)
    else:
        return sentences, sentence_ids, aspect_category_sentiments, (idx2aspectlabel, idx2sentilabel)

