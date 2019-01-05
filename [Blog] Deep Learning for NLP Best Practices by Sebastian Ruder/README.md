# Introduction

This [post](http://ruder.io/deep-learning-nlp-best-practices/) is a collection of best practices for using neural networks in Natural Language Processing. It will be updated periodically as new insights become available and in order to keep track of our evolving understanding of Deep Learning for NLP.

There has been a [running joke](https://twitter.com/IAugenstein/status/710837374473920512) in the NLP community that an LSTM with attention will yield state-of-the-art performance on any task. While this has been true over the course of the last two years, the NLP community is slowly moving away from this now standard baseline and towards more interesting models. However, we as a community do not want to spend the next two years independently (re-)discovering the *next* LSTM with attention. We do not want to reinvent tricks or methods that have already been shown to work. While many existing Deep Learning libraries already encode best practices for working with neural networks in general, such as initialization schemes, many other details, particularly task or domain-specific considerations, are left to the practitioner. The main goal of this article is to get you up to speed with the relevant best practices so you can make meaningful contributions as soon as possible.

This post is not meant to keep track of the state-of-the-art, but rather to collect best practices that are relevant for a wide range of tasks. In other words, rather than describing one particular architecture, this post aims to collect the features that underly successful architectures. While many of these features will be most useful for pushing the state-of-the-art, I hope that wider knowledge of them will lead to stronger evaluations, more meaningful comparison to baselines, and inspiration by shaping our intuition of what works.

# Best practices

## Word embeddings

Word embeddings are arguably the most widely known best practice in the recent history of NLP. It is well-known that using pre-trained embeddings helps (Kim, 2014).

The optimal dimensionality of word embeddings is mostly task-dependent: a smaller dimensionality works better for more **syntactic tasks** such as named entity recognition (Melamud et al., 2016) or part-of-speech (POS) tagging (Plank et al., 2016), while a larger dimensionality is more useful for more **semantic tasks** such as sentiment analysis (Ruder et al., 2016).

## Depth of Networks

While we will not reach the depths of computer vision for a while, neural networks in NLP have become progressively deeper. State-of-the-art approaches now regularly use deep Bi-LSTMs, typically consisting of 3-4 layers, e.g. for POS tagging (Plank et al., 2016) and semantic role labelling (He et al., 2017). Models for some tasks can be even deeper, cf. Google's NMT model with 8 encoder and 8 decoder layers (Wu et al., 2016). In most cases, however, performance improvements of making the model deeper than 2 layers are minimal (Reimers & Gurevych, 2017).

These observations hold for most sequence tagging and structured prediction problems. For classification, deep or very deep models perform well only with character-level input and shallow word-level models are still the state-of-the-art (Zhang et al., 2015; Conneau et al., 2016; Le et al., 2017).

## Layer connections

For training deep neural networks, some tricks are essential to avoid the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem). Different layers and connections have been proposed. Here, we will discuss three:

- highway layers
- residual connections
- dense connections

### Highway layers

Highway layers (Srivastava et al., 2015) are inspired by the gates of an LSTM. 

First let us assume a one-layer MLP, which applies an affine transformation followed by a non-linearity $g$ to its input $x$:
$$
\mathbf{h} = g(\mathbf{W}\mathbf{x} + \mathbf{b})
$$
A highway layer then computes the following function instead:
$$
\mathbf{h} = \mathbf{t} \odot g(\mathbf{W} \mathbf{x} + \mathbf{b}) + (1-\mathbf{t}) \odot \mathbf{x}
$$
where $\mathbf{t} = \sigma(\mathbf{W}_T \mathbf{x} + \mathbf{b}_T)$ is called the **transform gate**, and $(1-\mathbf{t})$ is called the **carry gate**. As we can see, highway layers are similar to the gates of an LSTM in that they adaptively *carry* some dimensions of the input directly to the output.

Highway layers have been used pre-dominantly to achieve state-of-the-art results for language modelling (Kim et al., 2016; Jozefowicz et al., 2016; Zilly et al., 2017), but have also been used for other tasks such as speech recognition (Zhang et al., 2016). [Sristava's page](http://people.idsia.ch/~rupesh/very_deep_learning/) contains more information and code regarding highway layers.

### Residual connections

Residual connections (He et al., 2016) have been first proposed for computer vision and were the main factor for winning ImageNet 2016. Residual connections are even more straightforward than highway layers and learn the following function:
$$
\mathbf{h} = g(\mathbf{W}\mathbf{x} + \mathbf{b}) + \mathbf{x}
$$
which simply adds the input of the current layer to its output via a short-cut connection. This simple modification mitigates the vanishing gradient problem, as the model can default to using the identity function if the layer is not beneficial.

### Dense connections

Rather than just adding layers from each layer to the next, dense connections (Huang et al., 2017) (best paper award at CVPR 2017) add direct connections from each layer to all subsequent layers. Dense connections then feed the concatenated output from all previous layers as input to the current layer:
$$
\mathbf{h}^l = g(\mathbf{W}[\mathbf{x}^1; \ldots; \mathbf{x}^l] + \mathbf{b})
$$
Dense connections have been successfully used in computer vision. They have also found to be useful for Multi-Task Learning of different NLP tasks (Ruder et al., 2017), while a residual variant that uses summation has been shown to consistently outperform residual connections for neural machine translation (Britz et al., 2017).

## Dropout regularizer

While **batch normalization** in computer vision has made other regularizers obsolete in most applications, dropout (Srivasta et al., 2014) is still the go-to regularizer for deep neural networks in NLP. A dropout rate of 0.5 has been shown to be effective in most scenarios (Kim, 2014). 

In recent years, variations of dropout such as adaptive (Ba & Frey, 2013) and evolutional dropout (Li et al., 2016) have been proposed, but none of these have found wide adoption in the community. The main problem hindering dropout in NLP has been that it could not be applied to recurrent connections, as the aggregating dropout masks would effectively zero out embeddings over time.

**Recurrent dropout** (Gal & Ghahramani, 2016) addresses this issue by applying the same dropout mask across timesteps at layer $l$. This avoids amplifying the dropout noise along the sequence and leads to effective regularization for sequence models. Recurrent dropout has been used for instance to achieve state-of-the-art results in semantic role labelling (He et al., 2017) and language modelling (Melis et al., 2017).

## Multi-task learning

If additional data is available, multi-task learning (MTL) can often be used to improve performance on the target task.

### Auxiliary objectives

We can often find auxiliary objectives that are useful for the task we care about (Ruder, 2017). While we can already predict surrounding words in order to pre-train word embeddings (Mikolov et al., 2013), we can also use this as an auxiliary objective during training (Rei, 2017). A similar objective has also been used by (Ramachandran et al., 2016) for sequence-to-sequence models.

### Task-specific layers

While the standard approach to MTL for NLP is hard parameter sharing, it is beneficial to allow the model to learn task-specific layers. This can be done by placing the output layer of one task at a lower level (Søgaard & Goldberg, 2016). Another way is to induce private and shared subspaces (Liu et al., 2017; Ruder et al., 2017).

