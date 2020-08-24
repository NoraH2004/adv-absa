# On the Robustness of a BERT Model against Input Level Adversarial Examples

Since their introduction in 2017, transformer-based language models took natural language processing by storm and are widely used in various applications, including question answering, machine translation, and text classification. One of the most used transformer models is BERT, which obtains state-of-the-art results in numerous benchmarks. Despite its popularity, the security vulnerability of the model against manipulated inputs existing in realistic scenarios is largely unknown, which is highly concerning given the increasing use in security-sensitive applications such as sentiment analysis and hate speech detection.
The term "adversarial examples" describes inputs crafted by adversaries with the intention of causing a deep neural network to change its classification output. Previous efforts have shown that transformer models are vulnerable to strategically designed adversarial examples in the white-box setting, a case where the model architecture and parameters are accessible to the adversary. However, black-box attacks pose a more realistic scenario. Furthermore, natural adversarial examples are more likely to occure in a real-world setting than intentionally crafted examples. 

This is the complementary code used in my Master's Thesis to explore the robustness of a BERT model used for the ABSA-task, a fine-grained sentiment analysis task that extracts both the aspect mentioned in a sentence and the sentiment associated with the aspect. The results of three different input level adversarial attacks in the black box setting indicate that the use of leet speak, misspellings, and additional punctuation marks has a strong impact on the model and alter the output. The attacks are conducted using the SemEval 2015 Dataset[[1]](#1) for ABSA and a TripAdvisor Hotel Review Dataset for Sentiment Analysis.

## Target Models

As the base for the experiments, we used the BERT base model [[2]](#2). In a first step, we pre-trained it on the laptop domain and fine-tuned it on the ABSA task in a second step. We employed the Amazon Laptop Reviews scraped by He et al. to have sufficient training data [[3]](#3) and used the Adam optimizer. We pre-train our model using a batch size of 32 and set the learning rate to 3 * 10^{-5} with random initialization.
We define a maximum input sequence length of 256 tokens, resulting in four sentences per sequence on average.  Due to the relatively low number of training data, we pre-train BERT base for 30 epochs, such that the model sees about 30 million sentences during training. That way, a single sentence appears multiple times within the two language model tasks. The code for pre-training can be found on [here](https://github.com/deepopinion/domain-adapted-atsc "Pre-Training on Laptop Domain").

For fine-tuning the pre-trained BERT model on the ABSA task, we used the Ranger optimizer and set the learning rate to 3 * 10^-5. We used a batch size of 32 and fine-tuned it for 20 epochs and random initialization. 

Moreover, this repo includes the code for attacking the [BERT base Mulitlingual Model](https://huggingface.co/bert-base-multilingual-uncased "BERT base Multilingual Uncased") for Sentiment Analysis, downlaoded from Huggingface.

## Adversarial Examples 

**Adversarial examples** are small, and often imperceptible perturbations applied to the input data in an effort to fool a deep classifier to incorrect classification [[4]](#4). These examples are a way to highlight model vulnerabilities and are useful for evaluation and interpretation of machine learning models. 
I generated adversarial text to attack a BERT model used for Sentiment Classification and ABSA by conducting non-targeted attacks in the black-box setting on the character level.

Using the **Leave One Out Method**[[5]](#5) for the 'important word' detection, I determine the word which has a critical influence on the model's prediction. I remove each word of a sentence one by one and let the model predict the incomplete sentences. 
Comparing the prediction before and after a word is removed reflects how the word influences the classification result. This procedure allows me to enhance the efficiency of my attacks.

To execute the perturbations, I focus on the input level rather than the embedding or semantic level. 

## Perturbations

I execute the attacks using three methods:

##### **133t 5p34k**
Testing the effect of [Leet Speak](https://en.wikipedia.org/wiki/Leet) on the BERT model
##### **Mispeelings**
Using the [Wikipedia List of Common Misspellings](https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings "Wikipedia List of common misspellings"), a list of common typos detected in Wikipedia articles. The file contains 4282 misspellings of 4282 words.
##### **,Punctuation.?**
Testing the influence of one additional comma after the important word.


## Results ABSA
|  | Leet Speak | Typos | Punctuation |
| -------- | ---------------------- | ------------------- | ------------------- |
|Total number of original sentences | 943 | 943 | 943
|Total number of modifyable original sentences | 897 | 369 | 943
|Total number of modifies sentences | 2232 | 1354 | 2555
|Total number of changed predictions through modification | 1066 | 420 | 382
|Modification Ratio | 47.76% | 31.02% | 14.95%


## Results Sentiment Analysis

|  | Leet Speak | Typos | Punctuation |
| -------- | ---------------------- | ------------------- | ------------------- |
|Size original Dataset | 435 | 435 | 435
|Size adversarial Dataset | 183 | 330 | 56
|Modification Ratio | 42.01% | 75.86% | 12.87%

## References
<a id="1">[1]</a>
Pontiki, M., Galanis, D., Papageorgiou, H., Manandhar, S., & Androutsopoulos, I. (2015).Semeval-2015 task 12: Aspect based sentiment analysis. In D. M. Cer, D. Jurgens,P. Nakov, & T. Zesch (Eds.),Proceedings of the 9th international workshop onsemantic evaluation,  semeval@naacl-hlt 2015,  denver,  colorado,  usa,  june 4-5,2015(pp. 486–495).  The Association for Computer Linguistics.  Retrieved fromhttps://doi.org/10.18653/v1/s15-2082doi: 10.18653/v1/s15-2082

<a id="2">[2]</a> 
Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2018b).  
    BERT: pre-training of deepbidirectional transformers for language understanding.CoRR,abs/1810.04805.
    Retrieved fromhttp://arxiv.org/abs/1810.04805

<a id="3">[3]</a> 
He, R., & McAuley, J. J. (2016). Ups and downs: Modeling the visual evolution of fashiontrends with one-class collaborative filtering. In J. Bourdeau, J. Hendler, R. Nkambou,I. Horrocks, & B. Y. Zhao (Eds.),Proceedings of the 25th international conference onworld wide web, WWW 2016, montreal, canada, april 11 - 15, 2016(pp. 507–517).ACM.   Retrieved fromhttps://doi.org/10.1145/2872427.2883037doi: 10.1145/2872427.2883037

<a id="4">[4]</a> 
Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I. J., & Fergus,R.  (2014).  Intriguing properties of neural networks.  In Y. Bengio & Y. LeCun(Eds.),2nd international conference on learning representations, ICLR 2014, banff,ab,  canada,  april  14-16,  2014,  conference  track  proceedings.Retrieved  fromhttp://arxiv.org/abs/1312.6199

<a id="5">[5]</a>
Jin, D., Jin, Z., Zhou, J. T., & Szolovits, P.  (2020).  Is BERT really robust?  A strongbaseline for natural language attack on text classification and entailment.  InThethirty-fourth AAAI conference on artificial intelligence, AAAI 2020, the thirty-secondinnovative applications of artificial intelligence conference, IAAI 2020, the tenthAAAI symposium on educational advances in artificial intelligence, EAAI 2020, newyork, ny, usa, february 7-12, 2020(pp. 8018–8025). AAAI Press. Retrieved fromhttps://aaai.org/ojs/index.php/AAAI/article/view/6311

