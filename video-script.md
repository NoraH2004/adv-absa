# Video Script ARES

## Adversarial Examples Against a BERT ABSA Model –
## Fooling Bert With L33T, Misspellign, and Punctuation,
-------------
This is the script for my video presentation of the ARES paper.
The texts in blockquotes are not in the script. I just did not want to delete them for good yet. (Just in case).

Currently I have 990 words excluding the results.
According to http://www.speechinminutes.com/ this leaves me 310 words for the results and other things that have to be added.

-------------
## Titlepage

Hello and welcome everyone! My name is Nora Hofer.  In the last year, I reasearched the robustness of a BERT Model for the ABSA task against adversarial attacks.

But what does all of that mean?

-------------
### BERT 

Let’s start with BERT. BERT is a transformer based language model introduced in 2018 by Gooogle Researches. Since then it is the SOTA transformer model applied in Natural Language Processing tasks. 

One of these tasks is called ABSA, which is an abbreviation for Aspact Based Sentiment Analysis. A fine-grained Sentiment Analysis that extracts both, the aspect mentioned in a sentence and the sentiment associated with these aspects. 


-------------
### Adversarial Examples Panda


Leaving us with Adversarial Attacks:
In 2013, the term adversarial example was introduced to describe scenarios where an adversary crafts inputs with the intention to cause a deep learning classifier to misclassify inputs. 

>Especially in the computer vision and image classification domain, powerful methods using gradient descent give the adversary the ability to mathematically optimize the attack and craft adversarial examples in the white-box setting, a setting where the model’s architecture and parameters are accessible to the adversary [3].

>The field of adversarial examples is related to the security of deep learning models.
Deep learning is influencing our lives in many ways and this will most likely continue and even increase in the coming years.


-------------
### Autonomous Driving

This phenomenon is highly concerning and very much related to security issues, for example, imagine deep learning in autonomous cars for recognizing road signs.  If the system is susceptible to small manipulations of the signs, the consequences could be fatal.

-------------
### Tweet (unmodified, unmarked)
These risks, however do not only occur in computer vision.  
>The increasing use of Deep Learning models for security-sensitive applications, such as
fake news and hate speech detection [13, 19], e.g. in social media
platforms and forums is alarming. 

In times of the COVID-19 pandemic, we once again see the importance of preventing the spread of misinformation on the Internet. 
This can be done, for example through a classification algorithm, as implemented by Twitter, that learns to detect untrustworthy information and labels the corresponding tweets to warn readers that they might be misinformed.

-------------
### Tweet (unmodified, marked)

Here you can see a Tweet containing misleading information about a cure for a Covid-19 Infection. The algorithm labels this tweet as critical and provides a link to further information. 

-------------
##### Tweet (modified, unmarked)
Through small modifications, in this case using leet speak in the word Covid, and replacing the letter o with the number 0, an attacker can theoretically circumvent the detector and recipients of the fake news would probably hardly notice afterwards that the sentence was deliberately manipulated.

There are tons of real-world scenarios where deep learning can and will be used. But when we apply it to security-critical areas, we need to know about their vulnerabilities and hence, security in terms of robustness against manipulations.

Our research investigates three attacks on the widely used BERT model for a classification task which could easily happen in a real-world setting. 

-------------
>Slide 7 - Outline

>In the following, I will quickly recap the differences in generating adversarial examples in computer vision and the text domain. 
I will then look into simple ways to fool a language model's prediction into generating perturbations inconspicuous to humans. 
Finally, I come to a conclusion and look forward to opening the discussion.

>Slide 8 – Bullet Points Differences CV - Text
(todo kürzen:)
Since textual data is discrete, we could not use gradient-based methods as applied in CV to generate them but had to use a different approach. 

>In an image, each pixel has a numerical representation within a fixed range and usually, pixels with similar numerical representations are closely related in terms of their characteristics.
As textual data is symbolic, it is not possible to apply the same logic to the text domain. Here, increasing or decreasing the numerical representation of a word or sentence by the value of one might alter the complete meaning of a sentence.

>Adversarial images, usually bounded by an 𝑙𝑝-norm, supposedly preserve the image’s semantic meaning. In the text domain changing a sentence’s semantic happens easily. The sentence “I like cats” could be changed to “I like dogs” by replacing a single word or to “I like cars” by replacing a single character, both times obviously changing its semantic.

>The thrid difference is that small modifications of image pixels are hardly recognized by human beings. Hence, if successful, adversarial examples will change the DNNs prediction but not human judgment. Small changes in the text, however, are easily detected and, therefore, render the possibility of attack failure. Moreover, a modification might also be corrected by spelling– or
grammar–check systems. It follows from these differences that, using gradient-based adversarial methods as in computer vision, for attacks in the text domain, can result in altered semantics, syntactically-incorrect sentences, or invalid words that cannot be matched with any words in the
word embedding space [5].
For that reason we generated adversarial samples in the black box setting following the following steps:


-------------
### Method Step 1
We started with training a BERT base model on the laptop domain and fine-tune it on the ABSA task in a second step. To do so we used the SemEval 2015 Task12, which is considered a benchmark dataset for research on the ABSA task. 

-------------
### Method Step 1 – ABSA – example sentence
Let’s see ABSA in action, using an example from the dataset:
The computer is excellent for gaming but I think it is way too expensive!!

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

We took each sentence and dropped one word after the other. After that each inclomplete sentece is predicted and the words, which changed the prediction through their absence, are stored seperately. 
In the example of our sentence - The important word is excellent.



-------------
### Method Step 3 – Modification

In the third step, we modify the identified important words to generate adversarial examples.

-------------
### Method Step 3 – Modification Methods

We designed three attack methods, namely Leet speak, misspellings and falsely placed punctuation marks.


>Different variations of word-, or character level based perturbations have been proposed in the literature. Examples include replacing, deleting, swapping, or inserting words or characters [16],
[5]. The disadvantage of word level perturbations is that an unintentional change of the semantic meaning of a sentence is likely to happen. This problem occurs less often when perturbing on the
character level since the perturbed words are likely to remain the same.


>By design, we have opted for perturbation methods on the character level since they are most likely do not alter an input sequence’s semantic meaning or grammar.
Even under circumstances where the change of a single character results in a new, existing word of the dictionary, with a different semantic meaning, we consider the example valid, since this scenario could also happen in a real world, e.g., as a typo. All our adversarial changes are supposed to prevent humans from easily spotting them. 

-------------
### Objectives 
As we believe that practical relevance is important for research in adversarial machine learning, we pursued the following objectives:
(1) keeping semantic meaning of the input data
(2) inconspicuousness to a human observer
(3) relevance in a real-world scenario.
>These are the main points in which our work differs from previous ones.

Let’s look into the attacks.

-------------
### Leet

The first one is leetspeak.

It is characterized by the use of non-alphabet characters to substitute one or multiple letters of one word with visually similar-looking symbols, so-called homoglyphs. We generate adversarial examples by swapping the letters a, e, l, o, and s of the identified important words with the numbers 4, 3, 1, 0, and 5, respectively.
>In online domains, humans do not find the usage of leetspeak suspicious. Thus, adversarial examples generated with this method will appear legitimate in those situations.

-------------
### Misspelling
The second one is misspellings. 
Inspired by [24], we use a list of common misspellings from Wikipedia to generate adversarial examples. We first determine the important words and then replace them with all possible misspellings. 
Also here, the semantic meaning of the modified word is preserved and the modification is unobtrusive to a reader. 

-------------
### Punctuation

The last one is the punctuation method. It is also the simplest one, where we wanted to find out whether a single comma added after the important word poses an efficient way to cause misclassifications. One additional comma might occur in practical use cases, and is not easily identified as an adversarial example by a human observer. 

-------------
### results todo.
>The efficiency of the attacks is measured using the success rate. A success rate of 100% would mean that any sentence containing an important word modified by one of my three methods caused BERT to predict an incorrect classification. 
A classification is considered incorrect if less or additional labels were predicted or the predicted labels change.
When using the Leet Speak method, we achieved a success rate of 47.8%, making it the most effective perturbation method.
The misspelling method was also very effective, which is even more surprising, considering that the misspellings dictionary that we used consisted of typos found in wikipedia articles and the BERT model is pre-trained on wikipedia corpus. 
It is important to note that the number of modifiable original sentences is quite low since there is not a typo representation for every important word.
With, as mentioned only adding one comma behind the determined important word, we were able to achieve a success rate of almost 15%. We chose this method for its high relevancesince many people have difficulties with English punctuation. 
Moreover, since the BERT Tokenizer separates punctuation marks from the preceding words, my method generates only one additional token but does not change the token representation of the important word. 


-------------
### Qualitative Results – Conclusion

Our experiments demonstrate that BERT can be fooled by input modifications on the character level, imitating real-world scenarios in the black-box setting.

The results of our three different attacks indicate that the use of leet speak, misspellings, and additional punctuation marks has a strong impact on the model and can alter the output. Compared to data augmentation, where semantics, grammar, and inconspicuousness are not a priority, we were able, through the careful design of the attack methods, to generate samples that are valid and do not change the human judgment, yet cause the classifier to produce false output labels.
This paper is inteded to raise awareness about the potential vulnerability of the BERT model and encourages to not entirely rely on these models for security relevant tasks, such as the detection of hate speech or false information.
Testing our generated adversarial datasets on other language models as a next step would provide information about the “transferability” of our attacks. Additionally, established countermeasures, such as adversarial training [18] should be further investigated for their effectiveness in
the text domain. Our result dataset can be used in the process in order to increase the robustness against such attacks. 

-------------
### FIN

Finally, we want to mention that we choose the title as it is intentionally, including the misspelling of “misspelling” and the comma at the very end, to highlight leetspeak, misspellings, and additional punctuation, the basis for our proposed attacks.

Please find more information and the code for our experiments, as well as the generated dataset in our github. 

Thanks.