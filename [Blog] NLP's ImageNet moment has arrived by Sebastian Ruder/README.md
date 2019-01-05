[TOC]



# Introduction

Big changes are underway in the world of Natural Language Processing (NLP). The long reign of word vectors as NLP’s core representation technique has seen an exciting new line of challengers emerge: [ELMo](https://arxiv.org/abs/1802.05365), [ULMFiT](https://arxiv.org/abs/1801.06146), and the [OpenAI transformer](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). These works made headlines ([1](https://blog.openai.com/language-unsupervised/) and [2](https://techcrunch.com/2018/06/15/machines-learn-language-better-by-using-a-deep-understanding-of-words/)) by demonstrating that **pretrained language models** can be used to achieve state-of-the-art results on a wide range of NLP tasks. Such methods herald a watershed moment: they may have the same wide-ranging impact on NLP as **pretrained ImageNet models** had on computer vision.

# From Shallow to Deep Pre-Training

## Shallow Pre-Training

Pretrained word vectors have brought NLP a long way. Proposed in 2013 as an approximation to language modeling, [word2vec](http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases) found adoption through its efficiency and ease of use in a time when hardware was a lot slower and deep learning models were not widely supported. Since then, the standard way of conducting NLP projects has largely remained unchanged: word embeddings pretrained on large amounts of unlabeled data via algorithms such as [word2vec](http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases) and [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) are used to initialize the first layer of a neural network, the rest of which is then trained on data of a particular task. 

On most tasks with limited amounts of training data, this led to a boost of [two to three percentage points](http://www.aclweb.org/anthology/D14-1181). Though these pretrained word embeddings have been immensely influential, they have a major limitation: they only incorporate previous knowledge in the first layer of the model---the rest of the network still needs to be trained from scratch.

Word2vec and related methods are *shallow* approaches that trade expressivity for efficiency. Using word embeddings is like initializing a computer vision model with pretrained representations that only encode edges: they will be helpful for many tasks, but they fail to capture higher-level information that might be even more useful. A model initialized with word embeddings needs to learn from scratch not only to disambiguate words, but also to derive meaning from a sequence of words. This is the core aspect of language understanding, and it requires modeling complex language phenomena such as compositionality, polysemy, anaphora, long-term dependencies, agreement, negation, and many more. It should thus come as no surprise that NLP models initialized with these **shallow representations** still require a huge number of examples to achieve good performance.

## Deep Pre-Training

At the core of the recent advances of ULMFiT, ELMo, and the OpenAI transformer is one key paradigm shift: going from just initializing the first layer of our models to pretraining the entire model with **hierarchical representations**. If learning word vectors is like only learning edges, these approaches are like learning the full hierarchy of features, from edges to shapes to high-level semantic concepts.

Interestingly, pretraining entire models to learn both low and high level features has been practiced for years by the computer vision (CV) community. Most often, this is done by learning to classify images on the large ImageNet dataset. ULMFiT, ELMo, and the OpenAI transformer have now brought the NLP community close to having an "ImageNet for language"---that is, a task that enables models to learn higher-level nuances of language, similarly to how ImageNet has enabled training of CV models that learn general-purpose features of images. In the rest of this piece, we’ll unpack just why these approaches seem so promising by extending and building on this analogy to ImageNet.

# Pretrained models for computer vision

## ImageNet and AlexNet

ImageNet’s impact on the course of machine learning research can hardly be overstated. The dataset was originally published in 2009 and quickly evolved into the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). In 2012, [the deep neural network](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) submitted by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton performed 41% better than the next best competitor, demonstrating that deep learning was a viable strategy for machine learning and arguably triggering the explosion of deep learning in ML research.

## Pretrained ImageNet models

The success of ImageNet highlighted that in the era of deep learning, data was at least as important as algorithms. Not only did the ImageNet dataset enable that very important 2012 demonstration of the power of deep learning, but it also allowed a breakthrough of similar importance in **transfer learning**: researchers soon realized that the weights learned in state of the art models for ImageNet could be used to initialize models for completely other datasets and improve performance significantly. This "fine-tuning" approach allowed achieving good performance with as little as one positive example per category (Donahue et al., 2014).

Pretrained ImageNet models have been used to achieve state-of-the-art results in tasks such as object detection, semantic segmentation, human pose estimation, and video recognition. At the same time, they have enabled the application of CV to domains where the number of training examples is small and annotation is expensive. Transfer learning via pretraining on ImageNet is in fact so effective in CV that not using it is now considered foolhardy (Mahajan et al., 2018).

## What makes ImageNet good for transfer learning

[Previous studies](https://arxiv.org/abs/1608.08614) have only shed partial light on this question: reducing the number of examples per class or the number of classes only results in a small performance drop, while fine-grained classes and more data are not always better. Rather than looking at the data directly, it is more prudent to probe what the models trained on the data learn. It is common knowledge that features of deep neural networks trained on ImageNet [transition from general to task-specific from the first to the last layer](https://arxiv.org/abs/1411.1792): lower layers learn to model low-level features such as edges, while higher layers model higher-level concepts such as patterns and entire parts or objects as can be seen in the figure below. Importantly, knowledge of edges, structures, and the visual composition of objects is relevant for many CV tasks, which sheds light on why these layers are transferred. A key property of an ImageNet-like dataset is thus to encourage a model to learn features that will likely generalize to new tasks in the problem domain.

Beyond this, it is difficult to make further generalizations about why transfer from ImageNet works quite so well. For instance, another possible advantage of the ImageNet dataset is the quality of the data. ImageNet’s creators went to great lengths to ensure reliable and consistent annotations. However, work in distant supervision serves as a counterpoint, indicating that large amounts of weakly labelled data might often be sufficient. In fact, recently researchers at Facebook showed that they could pre-train a model by [predicting hashtags on billions of social media images](https://arxiv.org/abs/1805.00932) to state-of-the-art accuracy on ImageNet.

Without any more concrete insights, we are left with two key desiderata:

1. An ImageNet-like dataset should be *sufficiently large*, i.e. on the order of millions of training examples.
2. It should be *representative of the problem space* of the discipline.

# Pretrained models for natural language

## NLP Tasks for Pre-Training

In NLP, models are typically a lot shallower than their CV counterparts. Analysis of features has thus mostly focused on the first embedding layer, and little work has investigated the properties of higher layers for transfer learning. Let us consider NLP tasks with the datasets that are large enough:

- **Reading comprehension** is the task of answering a natural language question about a paragraph. The most popular dataset for this task is the [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/), which contains more than 100,000 question-answer pairs and asks models to answer a question by highlighting a span in the paragraph.
- **Natural language inference** is the task of identifying the relation (entailment, contradiction, and neutral) that holds between a piece of text and a hypothesis. The most popular dataset for this task, the [Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/), contains 570k human-written English sentence pairs.
- **Machine translation**, translating text in one language to text in another language, is one of the most studied tasks in NLP, and---over the years---has accumulated vast amounts of training data for popular language pairs, e.g. 40M English-French sentence pairs in WMT 2014.
- **Constituency parsing** seeks to extract the syntactic structure of a sentence in the form of a (linearized) constituency parse tree. In the past, [millions of weakly labelled parses](https://arxiv.org/abs/1412.7449) have been used for training sequence-to-sequence models for this task.
- **Language modeling (LM)** aims to predict the next word given its previous word. Existing benchmark datasets consist of up to [1B words](http://www.statmt.org/lm-benchmark/), but as the task is unsupervised, any number of words can be used for training. 

All of these tasks provide---or would allow the collection of---a sufficient number of examples for training. Indeed, the above tasks (and many others such as sentiment analysis, constituency parsing , skip-thoughts, and autoencoding) have been used to pretrain representations in recent months.

## Challenges

While any data contains [some bias](https://arxiv.org/abs/1607.06520), human annotators may inadvertently introduce additional signals that a model can exploit. [Recent](https://www.aclweb.org/anthology/P16-1223) [studies](https://arxiv.org/abs/1803.02324) reveal that state-of-the-art models for tasks such as reading comprehension and natural language inference do not in fact exhibit deep natural language understanding but pick up on such cues to perform superficial pattern matching. For instance, [Gururangan et al. ](https://arxiv.org/abs/1803.02324) show that annotators tend to produce entailment examples simply by removing gender or number information and generate contradictions by introducing negations. A model that simply exploits these cues can correctly classify the hypothesis without looking at the premise in about 67% of the SNLI dataset.

The more difficult question thus is: Which task is most representative of the space of NLP problems? In other words, which task allows us to learn most of the knowledge or relations required for understanding natural language?

## Language Modelling

### The perfect task for pre-training

In order to predict the most probable next word in a sentence, a model is required not only to be able to express syntax (the grammatical form of the predicted word must match its modifier or verb) but also model semantics. Even more, the most accurate models must incorporate what could be considered *world knowledge* or *common sense*. Consider the incomplete sentence "The service was poor, but the food was". In order to predict the succeeding word such as “yummy” or “delicious”, the model must not only memorize what attributes are used to describe food, but also be able to identify that the conjunction “but” introduces a contrast, so that the new attribute has the opposing sentiment of “poor”.

Language modelling, the last approach mentioned, has been shown to capture many facets of language relevant for downstream tasks, such as [long-term dependencies](https://arxiv.org/abs/1611.01368), [hierarchical relations](https://arxiv.org/abs/1803.11138), and [sentiment](https://arxiv.org/abs/1704.01444). Compared to related unsupervised tasks such as skip-thoughts and autoencoding, [language modelling performs better on syntactic tasks even with less training data](https://openreview.net/forum?id=BJeYYeaVJ7).

Among the biggest benefits of language modelling is that training data comes for free with any text corpus and that potentially unlimited amounts of training data are available. This is particularly significant, as NLP deals not only with the English language. More than 4,500 languages are spoken around the world by more than 1,000 speakers. Language modeling as a pretraining task opens the door to developing models for previously underserved languages. For very low-resource languages where even unlabeled data is scarce, multilingual language models may be trained on multiple related languages at once, analogous to work on [cross-lingual embeddings](https://arxiv.org/abs/1706.04902).

### Methods: ELMo, ULMFiT and Open AI Transformer

So far, our argument for language modeling as a pretraining task has been purely conceptual. Pretraining a language model was [first proposed in 2015](https://arxiv.org/abs/1511.01432), but it remained unclear whether a single pretrained language model was useful for many tasks. In recent months, we finally obtained overwhelming empirical proof: [Embeddings from Language Models (ELMo)](https://arxiv.org/abs/1802.05365), [Universal Language Model Fine-tuning (ULMFiT)](https://arxiv.org/abs/1801.06146), and the [OpenAI Transformer](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) have empirically demonstrated how language modeling can be used for pretraining. All three methods employed pretrained language models to achieve state-of-the-art on a diverse range of tasks in Natural Language Processing, including text classification, question answering, natural language inference, coreference resolution, sequence labeling, and many others.

In many cases such as with ELMo, these improvements ranged between 10-20% better than the state-of-the-art on widely studied benchmarks, all with the single core method of leveraging a pretrained language model. ELMo furthermore won the best [paper award at NAACL-HLT 2018](https://naacl2018.wordpress.com/2018/04/11/outstanding-papers/), one of the top conferences in the field. Finally, these models have been shown to be extremely sample-efficient, achieving good performance with only hundreds of examples and are even able to perform zero-shot learning.

In light of this step change, **it is very likely that in a year’s time NLP practitioners will download pretrained language models rather than pretrained word embeddings** for use in their own models, similarly to how pre-trained ImageNet models are the starting point for most CV projects nowadays.

### Transfer learning

One outstanding question is how to transfer the information from a pre-trained language model to a downstream task. The two main paradigms for this are whether to use the pre-trained language model as a fixed feature extractor and incorporate its representation as features into a randomly initialized model as used in [ELMo](https://arxiv.org/abs/1802.05365), or whether to fine-tune the entire language model as done by [ULMFiT](https://arxiv.org/abs/1801.06146). The latter fine-tuning approach is what is typically done in CV where either the top-most or [several of the top layers are fine-tuned](https://arxiv.org/abs/1310.1531). While NLP models are typically more shallow and thus require different fine-tuning techniques than their vision counterparts, recent pretrained models are getting deeper. The next months will show the impact of each of the core components of transfer learning for NLP: an expressive language model encoder such as a deep BiLSTM or the [Transformer](https://arxiv.org/abs/1706.03762), the amount and nature of the data used for pretraining, and the method used to fine-tune the pretrained model.

The time is ripe for practical transfer learning to make inroads into NLP. In light of the impressive empirical results of [ELMo](https://arxiv.org/abs/1802.05365), [ULMFiT](https://arxiv.org/abs/1801.06146), and [OpenAI](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) it only seems to be a question of time until pretrained word embeddings will be dethroned and replaced by pretrained language models in the toolbox of every NLP practitioner. This will likely open many new applications for NLP in settings with limited amounts of labeled data. 

### Limitations

However, similar to word2vec, the task of language modeling naturally has its own limitations: It is only a proxy to true language understanding, and a single monolithic model is ill-equipped to capture the required information for certain downstream tasks. For instance, in order to answer questions about or follow the trajectory of characters in a story, a model needs to learn to perform [anaphora](https://en.wikipedia.org/wiki/Anaphora_(linguistics)) or [coreference resolution](https://en.wikipedia.org/wiki/Coreference#Coreference_resolution). In addition, language models can only capture what they have seen. Certain types of information, such as most common sense knowledge, are [difficult to learn from text alone](https://arxiv.org/abs/1705.11168) and require incorporating external information.

# But where’s the theory of pre-training?

Our analysis thus far has been mostly conceptual and empirical, as it is still poorly understood why models trained on ImageNet---and consequently on language modeling---transfer so well. One way to think about the generalization behaviour of pretrained models more formally is under a model of *bias learning* (Baxter, 2000). Assume our problem domain covers all permutations of tasks in a particular discipline, e.g. computer vision, which forms our *environment*. We are provided with a number of datasets that allow us to induce a *family* of hypothesis spaces. Our goal in bias learning is to find a *bias*, i.e. a hypothesis space that maximizes performance on the entire (potentially infinite) environment.

Empirical and theoretical results in multi-task learning (Caruana, 1997; Baxter, 2000) indicate that a bias that is learned on *sufficiently many* tasks is likely to generalize to unseen tasks drawn from the same environment. Viewed through the lens of multi-task learning, a model trained on ImageNet learns a large number of binary classification tasks (one for each class). These tasks, all drawn from the space of natural, real-world images, are likely to be representative of many other CV tasks. In the same vein, a language model---by learning a large number of classification tasks, one for each word---induces representations that are likely helpful for many other tasks in the realm of natural language. Still, much more research is necessary to gain a better theoretical understanding why language modeling seems to work so well for transfer learning.

