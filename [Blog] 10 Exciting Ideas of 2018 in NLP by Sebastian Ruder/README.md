# 1) Unsupervised machine translation

There were [two](https://arxiv.org/abs/1711.00043) [unsupervised](https://arxiv.org/abs/1710.11041) MT papers at ICLR 2018. They were *surprising* in that they worked at all, but results were still low compared to supervised systems. At EMNLP 2018, unsupervised MT hit its stride with [two](https://arxiv.org/abs/1804.07755) [papers](https://arxiv.org/abs/1809.01272) from the same two groups that significantly improve upon their previous methods.

# 2) Pretrained language models

Using pretrained language models is probably the most significant NLP trend in 2018. There have been a slew of memorable approaches: [ELMo](https://arxiv.org/abs/1802.05365), [ULMFiT](https://arxiv.org/abs/1801.06146), [OpenAI Transformer](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), and [BERT](https://arxiv.org/abs/1810.04805).

# 3) Common sense inference datasets

Incorporating common sense into our models is one of the most important directions moving forward. However, creating good datasets is not easy and even popular ones [show](http://aclweb.org/anthology/N18-2017) [large](http://www.aclweb.org/anthology/S18-2023) biases. This year, there have been some well-executed datasets that seek to teach models some common sense such as [Event2Mind](https://arxiv.org/abs/1805.06939) and [SWAG](https://arxiv.org/abs/1808.05326), both from the University of Washington.

[Visual Commonsense Reasoning](http://visualcommonsense.com/) is the first visual QA dataset that includes a rationale (an explantation) with each answer.

# 4) Meta-learning

Meta-learning has seen much use in few-shot learning, reinforcement learning, and robotics—the most prominent example: [model-agnostic meta-learning (MAML)](https://arxiv.org/abs/1703.03400)—but successful applications in NLP have been rare. Meta-learning is most useful for problems with a limited number of training examples.

[Meta-Learning for Low-Resource Neural Machine Translation(EMNLP 2018)](http://aclweb.org/anthology/D18-1398): The authors use MAML to learn a good initialization for translation, treating each language pair as a separate meta-task. Adapting to low-resource languages is probably the most useful setting for meta-learning in NLP. In particular, combining multilingual transfer learning (such as [multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md)), unsupervised learning, and meta-learning is a promising direction.

# 5) Robust unsupervised methods

This year, [we](http://aclweb.org/anthology/P18-1072) [and](http://aclweb.org/anthology/D18-1056) others have observed that unsupervised cross-lingual word embedding methods break down when languages are dissimilar. This is a common phenomenon in transfer learning where a discrepancy between source and target settings (e.g. domains in [domain adaptation](https://www.cs.jhu.edu/~mdredze/publications/sentiment_acl07.pdf), tasks in [continual learning](https://arxiv.org/abs/1706.08840) and [multi-task learning](http://www.aclweb.org/anthology/E17-1005)) leads to deterioration or failure of the model. Making models more robust to such changes is thus important.

# 6) Understanding representations

There have been a lot of efforts in better understanding representations. In particular, ['diagnostic classifiers'](https://arxiv.org/abs/1608.04207) (tasks that aim to measure if learned representations can predict certain attributes) have become [quite common](http://arxiv.org/abs/1805.01070).

[Dissecting Contextual Word Embeddings: Architecture and Representation (EMNLP 2018)](http://aclweb.org/anthology/D18-1179): This paper does a great job of better understanding pretrained language model representations.

# 7) Clever auxiliary tasks

In many settings, we have seen an increasing usage of multi-task learning with carefully chosen auxiliary tasks. For a good auxiliary task, data must be easily accessible. One of the most prominent examples is [BERT](https://arxiv.org/abs/1810.04805), which uses next-sentence prediction (that has been used in [Skip-thoughts](https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf) and more recently in [Quick-thoughts](https://arxiv.org/pdf/1803.02893.pdf)) to great effect.

- [**Syntactic Scaffolds for Semantic Structures** (EMNLP 2018)](http://aclweb.org/anthology/D18-1412): This paper proposes an auxiliary task that pretrains span representations by predicting for each span the corresponding syntactic constituent type. Despite being conceptually simple, the auxiliary task leads to large improvements on span-level prediction tasks such as semantic role labelling and coreference resolution. This papers shows that specialised representations learned at the level required by the target task (here: spans) are immensely beneficial.
- [**pair2vec: Compositional Word-Pair Embeddings for Cross-Sentence Inference**](https://arxiv.org/abs/1810.08854): In a similar vein, this paper pretrains *word pair representations* by maximizing the pointwise mutual information of pairs of words with their context. This encourages the model to learn more meaningful representations of word pairs than with more general objectives, such as language modelling. The pretrained representations are effective in tasks such as SQuAD and MultiNLI that require cross-sentence inference. We can expect to see more pretraining tasks that capture properties particularly suited to certain downstream tasks and are complementary to more general-purpose tasks like language modelling.

# 8) Combining semi-supervised learning with transfer learning

With the recent advances in transfer learning, we should not forget more explicit ways of using target task-specific data. In fact, pretrained representations are complementary with many forms of semi-supervised learning. We have explored [self-labelling approaches](http://aclweb.org/anthology/P18-1096), a particular category of semi-supervised learning.

[**Semi-Supervised Sequence Modeling with Cross-View Training**(EMNLP 2018)](http://aclweb.org/anthology/D18-1217): This paper shows that a conceptually very simple idea, making sure that the predictions on different views of the input agree with the prediction of the main model, can lead to gains on a diverse set of tasks.

# 9) QA and reasoning with large documents

There have been a lot of developments in question answering (QA), with an [array](https://arxiv.org/abs/1809.09600) [of](https://stanfordnlp.github.io/coqa/)[new](http://quac.ai/) [QA](https://arxiv.org/abs/1806.03822) [datasets](http://qangaroo.cs.ucl.ac.uk/). Besides conversational QA and performing multi-step reasoning, the most challenging aspect of QA is to synthesize narratives and large bodies of information.

[**The NarrativeQA Reading Comprehension Challenge** (TACL 2018)](http://aclweb.org/anthology/Q18-1023): This paper proposes a challenging new QA dataset based on answering questions about entire movie scripts and books.

# 10) Inductive bias

Inductive biases such as convolutions in a CNN, regularization, dropout, and other mechanisms are core parts of neural network models that act as a regularizer and make models more sample-efficient. However, coming up with a broadly useful inductive bias and incorporating it into a model is challenging.

[**Sequence classification with human attention** (CoNLL 2018)](http://aclweb.org/anthology/K18-1030): This paper proposes to use human attention from eye-tracking corpora to regularize attention in RNNs.

[**Linguistically-Informed Self-Attention for Semantic Role Labeling**(EMNLP 2018)](http://aclweb.org/anthology/D18-1548): This paper has a lot to like: a Transformer trained jointly on both syntactic and semantic tasks; the ability to inject high-quality parses at test time; and out-of-domain evaluation.

