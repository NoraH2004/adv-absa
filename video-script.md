# Video Script ARES

## Adversarial Examples Against a BERT ABSA Model –
## Fooling Bert With L33T, Misspellign, and Punctuation,
-------------
This is the script for my video presentation of the ARES paper.
The texts in blockquotes are not in the script. I just did not want to delete them for good yet. (Just in case).

Currently I have 1748 words.
According to http://www.speechinminutes.com/ this makes 10.9 Minutes, when talking fast. Let's see :P

-------------
## Titlepage


Hello and welcome everyone! My name is Nora Hofer.  In the last year, we researched the robustness of a BERT Model for the ABSA task against adversarial attacks.

But what does all of that mean?

-------------
### BERT 

Let’s start with BERT. BERT is a transformer-based language model introduced in 2018 by Google researchers. Since then it is the state-of-the-art model applied in Natural Language Processing tasks. 

One of these tasks is called ABSA, which is an abbreviation for Aspect Based Sentiment Analysis. A fine-grained Sentiment Analysis that extracts both, the aspect mentioned in a sentence and the sentiment associated with these aspects. 


-------------
### Adversarial Examples Panda
Leaving us with Adversarial Attacks:
Adversarial Examples are minimally altered, carefully crafted inputs with the intention to cause a model to misclassify inputs.


-------------
### Autonomous Driving

This phenomenon is highly concerning and very much related to security issues, for example, imagine deep learning in autonomous cars for recognizing road signs.  If the system is susceptible to small manipulations of the signs, the consequences could be fatal.

-------------
### Tweet (unmodified, unmarked)
These risks, however, do not only occur in computer vision.  


In times of the COVID-19 pandemic, we once again see the importance of preventing the spread of misinformation on the Internet. 
This can be done, for example through a classification algorithm, as implemented by Twitter, that learns to detect untrustworthy information and labels the corresponding tweets to warn readers that they might be misinformed.

-------------
### Tweet (unmodified, marked)

Here you can see a Tweet containing misleading information about a cure for a Covid-19 Infection. The algorithm labels this tweet as critical and provides a link to further information. 

-------------
##### Tweet (modified, unmarked)
Through small modifications, in this case, using leet speak in the word Covid, and replacing the letter o with the number 0, an attacker can theoretically circumvent the detector, and recipients of the fake news would probably hardly notice afterwards that the sentence was deliberately manipulated.

There are tons of real-world scenarios where deep learning can and will be used. But when we apply it to security-critical areas, we need to know about their vulnerabilities and hence, security in terms of robustness against manipulations.

Our research investigates three attacks on the widely used BERT model for a classification task that could easily happen in a real-world setting. 

Since textual data is discrete, we could not use gradient-based methods as applied in computer vision to generate adversarial examples but had to use a different approach. 

Let's get into it.
-------------
### Method Step 1
We started off with training a BERT base model on the laptop domain ... 


-------------
### Method Step 1 – ABSA 
... and fine-tune it on the ABSA task in a second step. 


-------------
### Method Step 1 – SemEval 2015 

To do so we used the SemEval 2015 Task 12, which is considered a benchmark dataset for research on the ABSA task. 

An aspect category is defined as a combination of an entity and an attribute describing the entity. The second part of the annotation is the sentiment label, which expresses the polarity towards the aspect category and can take on the values positive, neutral, and negative. 

-------------
### Method Step 1 – ABSA – Example Sentence
Let’s see ABSA in action, using an example from the dataset:
*The computer is excellent for gaming but I think it is way too expensive!!*

-------------
### Method Step 1 – ABSA – Example classified
In this case, the model identifies the aspects gaming (having a positive sentiment) and price (with a negative sentiment).

-------------
### Method Step 2 – Important Word
In the second step, we used the trained ABSA Model to identify the word of a sentence, which is important for the model’s prediction.

-------------
### Method Step 2 - LOO
To do so we used a method called Leave-One-Out Method.

-------------
### Method Step 2 – Sentece Words dropped & classified

We took each sentence and dropped one word after the other. After that, each incomplete sentence is predicted and the words, which changed the prediction through their absence, are stored separately. 
In the example of our sentence - The only important word is *excellent*.



-------------
### Method Step 3 – Modification

In the third step, we modify the identified important words to generate adversarial examples.

-------------
### Method Step 3 – Modification Methods

We designed three attack methods, namely Leetspeak, misspellings, and falsely placed punctuation marks.


Different variations of word-, or character level-based perturbations have been proposed in the literature. Examples include replacing, deleting, swapping, or inserting words or characters. By design, we have opted for perturbation methods on the character level since they most likely do not alter an input sequence’s semantic meaning or grammar.

All our adversarial changes are supposed to prevent humans from easily spotting them. 

-------------
### Objectives 
As we believe that practical relevance is important for research in adversarial machine learning, we pursued the following objectives:
(1) keeping semantic meaning of the input data
(2) inconspicuousness to a human observer
(3) relevance in a real-world scenario.


Let’s look into the attacks.

-------------
### Leet

The first one is leetspeak.

It is characterized by the use of non-alphabet characters to substitute one or multiple letters of one word with visually similar-looking symbols, so-called homoglyphs. We generate adversarial examples by swapping the letters a, e, l, o, and s of the identified important words with the numbers 4, 3, 1, 0, and 5, respectively.

See here the resulting adversarial example sentence after the modification, which causes the classifier to switch from Gaming - positive to Gaming - negative.


-------------
### Misspelling
The second one is misspellings. 
Inspired by [24], we use a list of common misspellings from Wikipedia to generate adversarial examples. After determining the important words we replace them with all possible misspellings. 
Also here, the semantic meaning of the modified word is preserved and the modification is unobtrusive to a reader. 

Through the modification, the classifier is no longer able to detect the positive sentiment towards the aspect gaming. 

-------------
### Punctuation

The last one is the punctuation method. It is also the simplest one, where we wanted to find out whether a single comma added after the important word poses an efficient way to cause misclassifications. One additional comma might occur in practical use cases and is not easily identified as an adversarial example by a human observer. 

Here the classifier falsely detects a negative sentiment about the Laptop in general. 

-------------
### Results 
Let's look at the results for the whole dataset. 

The efficiency of our attacks is measured using a distinct and an overall success rate. 
The scores are calculated as follows:

To generate Dataset A, we filtered the SemEval 2015 dataset for unique items, resulting in 943 sentences.
We then use the LOO method to detect important words for all sentences in Dataset A and check if any of the important words can be modified, resulting in Dataset B. As you can see, for the punctuation method, by nature every sentence is modifiable. 
In the next step, we created all possible potential adversarial examples from Dataset B, for example, all possible misspellings for all identified important words, which we call Dataset C.
Remember, that one sentence can potentially have more than one important word. 

Finally, we use the BERT model to predict the aspect and sentiment of all elements from Dataset C and compare the predictions to the original one. 

A classification is considered incorrect if fewer or additional labels were predicted or at least one of the predicted labels change.

Dataset D contains all sentences from Dataset C, that were changed in one of the mentioned ways. 
Dataset E contains all sentences from Dataset B, that were changed in one of the mentioned ways. That means, if per sentence, one modification caused the prediction to change, the sentence is counted as adversarial and added to dataset E.


The overall success rate is calculated as the ratio Dataset D / Dataset C. 
And the distinct success rate is calculated as the ratio Dataset E / Dataset B.

An overall success rate of 100% would mean that any sentence containing an important word modified by one of my three methods caused BERT to predict an incorrect classification. 


When using leetspeak, we achieved an overall success rate of 47.8% and a distinct success rate of 88% making it the most effective perturbation method.
The misspelling method was also very effective, which is even more surprising, considering that the misspellings dictionary that we used consisted of typos found in Wikipedia articles and the BERT model is pre-trained on Wikipedia corpus. We achieved an overall, and a distinct success rate of 31, and 70% respectively.
With, as mentioned only adding one comma behind the determined important word, we were able to achieve an overall success rate of almost 15% and a distinct success rate of 27%. 

For the case of our example sentence "It's wonderful for computer gaming" we were able to change the result of the prediction by using any of the three methods.





-------------
### Conclusion & Further Steps

Let's conclude.


DNN-based text classification continuously gains importance for enhancing the safety of users, for example in online forums or social media, where leetspeak is commonly used. 

Our experiments demonstrate that BERT-based ABSA models can be fooled by input modifications on the character level, imitating real-world scenarios in the black-box setting.

We were able to generate samples that are valid and do not change human judgment, yet cause the classifier to produce false output labels.
The paper is intended to raise awareness about the potential vulnerability of the BERT model and encourages us to not entirely rely on these models for security-relevant tasks, such as the detection of hate speech or false information.
Testing our generated adversarial datasets on other language models as a next step would provide information about the “transferability” of our attacks. Additionally, established countermeasures, such as adversarial training should be further investigated for their effectiveness in
the text-domain. Our result dataset can be used in the process to increase the robustness against such attacks. 

-------------
### FIN

Finally, we want to mention that we choose the title as it is intentionally, including the modifications and misspelling, to highlight our proposed attacks.

If you liked this presentation and are interested, I recommend reading the paper and/or looking into our GitHub to find the code, as well as the generated adversarial datasets.
Also, do not hesitate to reach out, if you have any questions or comments.

Thank you for listening.