# Introduction

Over the course of this blog post, I will first introduce **GloVe**, a popular word embedding model. I will then highlight the connection between word embedding models and distributional semantic methods. Subsequently, I will introduce the four models that will be used to measure the impact of the different factors. I will then give an overview of all additional factors that play a role in learning word representations, besides the choice of the algorithm.

In this post, I want to highlight the factors, i.e. the secret ingredients that account for the success of word2vec. In particular, I want to focus on the connection between word embeddings trained via neural models and those produced by traditional **distributional semantics models** (DSMs). By showing how these ingredients can be transferred to DSMs, I will demonstrate that distributional methods are in no way inferior to the popular word embedding methods. Even though this is no new insight, I feel that traditional methods are frequently overshadowed amid the deep learning craze and their relevancy consequently deserves to be mentioned more often.

# GloVe

Briefly, GloVe seeks to make explicit what the skip-gram with negative-sampling (*SGNS*) method does implicitly: Encoding meaning as vector offsets in an embedding space -- seemingly only a by-product of word2vec -- is the specified goal of GloVe.

Specifically, the authors of Glove show that the ratio of the co-occurrence probabilities of two words (rather than their co-occurrence probabilities themselves) is what contains information and aim to encode this information as vector differences. To achieve this, they propose a weighted least squares objective that directly aims to minimise the difference between the dot product of the vectors of two words and the logarithm of their number of co-occurrences:
$$
J = \sum\limits_{i, j=1}^V f(X_{ij})   (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \text{log}   X_{ij})^2
$$

where $X_{ij}$ is the number of times word $i$ occurs in the context of word $j$, and $f$ is a weighting function that assigns relatively lower weight to rare and frequent co-occurrences.

You can find some additional intuitions on GloVe and its difference to word2vec by the author of gensim [here](http://rare-technologies.com/making-sense-of-word2vec/), in [this Quora thread](https://www.quora.com/How-is-GloVe-different-from-word2vec), and in [this blog post](https://cran.r-project.org/web/packages/text2vec/vignettes/glove.html).

# Word embeddings vs. distributional semantics models (DSMs)

We can view DSMs as **count models** as they "count" co-occurrences among words by operating on co-occurrence matrices. In contrast, neural word embedding models can be seen as **predict models**, as they try to predict surrounding words.

The reason why word embedding models, particularly word2vec and GloVe, became so popular is that they seemed to continuously and significantly outperform DSMs. Many attributed this to the neural architecture of word2vec or the fact that it predicts words, which seemed to have a natural edge over solely relying on co-occurrence counts.

In 2014, Baroni et al. showed that predict models consistently outperform count models in almost all tasks, thus providing a clear verification for the apparent superiority of word embedding models. Is this the end? No.

Already with GloVe we've seen that the differences are not as clear-cut: While GloVe is considered a predict model by Levy et al. (2015), it is clearly factorizing a word-context co-occurrence matrix, which brings it close to traditional methods such as PCA and LSA. Even more, Levy et al. demonstrate that word2vec implicitly factorizes a word-context PMI matrix.

Consequently, while on the surface DSMs and word embedding models use different algorithms to learn word representations -- the first count, the latter predict -- fundamentally, both types of models act on the same underlying statistics of the data, i.e. the co-occurrence counts between words.

Thus, the question that still remains and which we will dedicate the rest of this blog post to answering is the following:
*Why do word embedding models still perform better than DSM with almost the same information?*

# Models

Following Levy et al. (2015), we will isolate and identify the factors that account for the success of neural word embedding models and show how these can be transferred to traditional methods by comparing the following four models:

- **Positive Pointwise Mutual Information (PPMI)**: PMI is a common measure for the strength of association between two words. It is defined as the log ratio between the joint probability of two words $w$ and $c$ and the product of their marginal probabilities: $PMI(w,c) = \text{log}   \dfrac{P(w,c)}{P(w) P(c)}$. As if pairs $(w, c)$ were never observed, $PMI(w,c)=- \infty$, PMI is in practice often replaced with *positive* PMI (PPMI), which replaces negative values with: $PPMI(w,c) = \text{max}(PMI(w,c),0)$.
- **Singular Value Decomposition (SVD):** SVD is one of the most popular methods for dimensionality reduction and found its into NLP originally via latent semantic analysis (LSA). SVD factories the word-context co-occurrence matrix into the product of three matrices $U \cdot \Sigma \times V^T$. In practice, SVD is often used to factorize the matrix produced by PPMI. Generally, only the top $d$ elements of $\Sigma$ are kept, yielding $W^{SVD} = U_d \cdot \Sigma_d$ (word representations) and $C^{SVD} = V_d$ (context representations).
- **Skip-gram with Negative Sampling (SGNS)** aka word2vec
- **Global Vectors (GloVe)**

# Hyperparameters

We will look at the following hyper-parameters:

- Pre-processing
  - Dynamic context window
  - Subsampling frequent words
  - Deleting rare words
- Association metric
  - Shifted PMI
  - Context distribution smoothing
- Post-processing
  - 

## Pre-processing

Word2vec introduces three ways of pre-processing a corpus, which can be easily applied to DSMs.

### Dynamic context window

In DSMs traditionally, the context window is unweighted and of a constant size. Both SGNS and GloVe, however, use a scheme that assigns more weight to closer words, as closer words are generally considered to be more important to a word's meaning. Additionally, in SGNS, the window size is not fixed, but the actual window size is dynamic and sampled uniformly between 1 and the maximum window size during training.

### Subsampling frequent words

SGNS dilutes very frequent words by randomly removing words whose frequency $f$ is higher than some threshold $t$ with a probability $p = 1 - \sqrt{\dfrac{t}{f}}$. As this subsampling is done *before* actually creating the windows, the context windows used by SGNS in practice are larger than indicated by the context window size.

### Deleting rare words

In the pre-processing of SGNS, rare words are also deleted *before* creating the context windows, which increases the actual size of the context windows further. Levy et al. (2015) find this not to have a significant performance impact, though.

## Association metric

PMI has been shown to be an effective metric for measuring the association between words. Since Levy and Goldberg (2014) have shown SGNS to implicitly factorize a PMI matrix, two variations stemming from this formulation can be introduced to regular PMI.

### Shifted PMI

In SGNS, the higher the number of negative samples $k$, the more data is being used and the better should be the estimation of the parameters. $k$ affects the shift of the PMI matrix that is implicitly factorized by word2vec, i.e.  k shifts the PMI values by $\log k$.
$$
SPPMI(w,c) = \text{max}(PMI(w,c) - \text{log}   k,0)
$$

### Context distribution smoothing

In SGNS, the negative samples are sampled according to a *smoothed* unigram distribution, i.e. an unigram distribution raised to the power of $\alpha$, which is empirically set to $\dfrac 3 4$. This leads to frequent words being sampled relatively less often than their frequency would indicate.

We can transfer this to PMI by equally raising the frequency of the context words $f(c)$ to the power of $\alpha$:
$$
PMI(w, c) = \text{log} \dfrac{p(w,c)}{p(w)p_\alpha(c)}
$$
where $p_\alpha(c) = \dfrac{f(c)^\alpha}{\sum_c f(c)^\alpha}$

## Post-processing

Similar as in pre-processing, three methods can be used to modify the word vectors produced by an algorithm.

### Adding context vectors

The authors of GloVe propose to add word vectors and context vectors to create the final output vectors, e.g. $\vec{v}_{\text{cat}} = \vec{w}_{\text{cat}} + \vec{c}_{\text{cat}}$. However, this method cannot be applied to PMI, as the vectors produced by PMI are sparse.

### Eigenvalue weighting

SVD produces the following matrices: $W^{SVD} = U_d \cdot \Sigma_d$ (non-orthonormal) and $C^{SVD} = V_d$ (orthonormal). 

### Vector normalisation

Finally, we can also normalise all vectors to unit length.

# Results

Levy et al. (2015) train all models on a dump of the English wikipedia and evaluate them on the commonly used word similarity and analogy datasets.

Levy et al. find that SVD -- and not one of the word embedding algorithms -- performs best on similarity tasks, while SGNS performs best on analogy datasets. They furthermore shed light on the importance of hyperparameters compared to other choices:

1. Hyperparameters vs. algorithms:
   Hyperparameter settings are often more important than algorithm choice.
   No single algorithm consistently outperforms the other methods.
2. Hyperparameters vs. more data:
   Training on a larger corpus helps for some tasks.
   In 3 out of 6 cases, tuning hyperparameters is more beneficial.

# Conclusion

Equipped with these insights, we can now debunk some generally held claims:

1. Are embeddings superior to distributional methods?
   With the right hyperparameters, no approach has a consistent advantage over another.
2. Is GloVe superior to SGNS?
   SGNS outperforms GloVe on all tasks.
3. Is CBOW a good word2vec configuration?
   CBOW does not outperform SGNS on any task.

Finally -- and one of the things I like most about the paper -- we can give concrete practical recommendations:

- **DON'T** use shifted PPMI with SVD.
- **DON'T** use SVD "correctly", i.e. without eigenvector weighting (performance drops 15 points compared to with eigenvalue weighting with p=0.5).
- **DO** use PPMI and SVD with short contexts (window size of 22).
- **DO** use many negative samples with SGNS.
- **DO** always use context distribution smoothing (raise unigram distribution to the power of α=0.75) for all methods.
- **DO** use SGNS as a baseline (robust, fast and cheap to train).
- **DO** try adding context vectors in SGNS and GloVe.

These results run counter to what is generally assumed, namely that word embeddings are superior to traditional methods and indicate that it generally makes *no difference whatsoever* whether you use word embeddings or distributional methods -- what matters is that you tune your hyperparameters and employ the appropriate pre-processing and post-processing steps.

Recent papers from Jurafsky's group echo these findings and show that SVD -- not SGNS -- is often the preferred choice when you care about accurate word representations.

I hope this blog post was useful in highlighting cool research that sheds light on the link between traditional distributional semantic and in-vogue embedding models. As we've seen, knowledge of distributional semantics allows us to improve upon our current methods and develop entirely new variations of existing ones. For this reason, I hope that the next time you train word embeddings, you will consider adding distributional methods to your toolbox or lean on them for inspiration.

