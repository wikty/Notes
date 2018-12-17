[TOC]



# Introduction

Sparse coding is a class of unsupervised methods for learning sets of over-complete bases to represent data efficiently. The aim of sparse coding is to find a set of basis vectors $\mathbf{\phi}_i$ such that we can represent an input vector $x \in \R^n$ as a linear combination of these basis vectors:
$$
\begin{align}
\mathbf{x} = \sum_{i=1}^k a_i \mathbf{\phi}_{i} 
\end{align}
$$
where $k>n$ means the basis vectors $\phi_i$ is an **over-complete set**, it must not be linearly independent; the coefficients $a_i$ is associated with each example $x$; the basis vectors $\phi_i$ is associated with all of examples.

For dataset $X=\{\mathbf{x_1}, \mathbf{x_2}, \dots, \mathbf{x_m}\}$, we get:
$$
\mathbf{x_j} = \Phi \mathbf{a_j}
\\
X = \Phi A
$$

$$
\begin{align}
X = 
\begin{bmatrix} 
| & | & & |  \\
\mathbf{x_1} & \mathbf{x_2} & \cdots & \mathbf{x_m}  \\
| & | & & | 
\end{bmatrix}       
\end{align}

\qquad \text{where} \quad x_i \in \R^n
$$

$$
\begin{align}
A = 
\begin{bmatrix} 
| & | & & |  \\
\mathbf{a_1} & \mathbf{a_2} & \cdots & \mathbf{a_m}  \\
| & | & & | 
\end{bmatrix}

\qquad \text{where} \quad a_i \in \R^k
\end{align}
$$

$$
\begin{align}
\Phi = 
\begin{bmatrix} 
| & | & & |  \\
\mathbf{\phi_1} & \mathbf{\phi_2} & \cdots & \mathbf{\phi_k}  \\
| & | & & | 
\end{bmatrix}

\qquad \text{where} \quad {\phi}_i \in \R^n
\end{align}
$$

# Motivation

While techniques such as Principal Component Analysis (PCA) allow us to learn a **complete set** of basis vectors efficiently, we wish to learn an **over-complete set** of basis vectors to represent input vectors. The advantage of having an over-complete basis is that our basis vectors are better able to capture structures and patterns inherent in the input data. 

However, with an over-complete basis, the coefficients $a_i​$ are no longer uniquely determined by the input vector $x​$. Therefore, in sparse coding, we introduce the additional criterion of **sparsity** to resolve the degeneracy introduced by over-completeness. Here, we define sparsity as having few non-zero components or having few components not close to zero. The requirement that our coefficients $a_i​$ be sparse means that given a input vector, we would like as few of our coefficients to be far from zero as possible. 

The choice of sparsity as a desired characteristic of our representation of the input data can be motivated by the observation that most sensory data such as *natural images may be described as the superposition of a small number of atomic elements such as surfaces or edges*.

# Optimization Objective

We define the sparse coding cost function on a set of $m$ input vectors as:
$$
\begin{align}
\mathbf{L} &= \sum_{j=1}^{m} \left( \mathbf{L}_{reconstruction}^{(j)} + \mathbf{L}_{penalty}^{(j)} \right)
\\
\mathbf{L}_{reconstruction}^{(j)} &= \left|\left| \mathbf{x}^{(j)} - \sum_{i=1}^k a^{(j)}_i \mathbf{\phi}_{i}\right|\right|^{2}
\\
\mathbf{L}_{penalty}^{(j)} &= \lambda \sum_{i=1}^{k}S(a^{(j)}_i)
\end{align}
$$
where $S(.)$ is a sparsity cost function which penalizes $a_i$ for being far from zero. the parameters of the cost function are $\phi_i$ and $a^{(j)}_i$ for all $i, j$.

Although the most direct measure of sparsity is the $L_0$ norm: $S(a_i) = \mathbf{1}(|a_i|>0)$, it is non-differentiable and difficult to optimize in general. In practice, common choices for the sparsity cost $S(.)$ are the $L_1$ penalty $S(a_i)=\left|a_i\right|_1$ and log penalty $S(a_i)=\log(1+a_i^2)$.

In addition, it is also possible to make the sparsity penalty arbitrarily small by scaling down $a_i$ and scaling $\phi_i$ up by some large constant. To prevent this from happening, we will constrain $||\phi_i||^2$ to be less than some constant $C$:
$$
\begin{array}{rc}
\text{minimize}_{a^{(j)}_i,\mathbf{\phi}_{i}} & \sum_{j=1}^{m} \left|\left| \mathbf{x}^{(j)} - \sum_{i=1}^k a^{(j)}_i \mathbf{\phi}_{i}\right|\right|^{2} + \lambda \sum_{i=1}^{k}S(a^{(j)}_i) 
\\
\text{subject to}  &  \left|\left|\mathbf{\phi}_i\right|\right|^2 \leq C, \forall i = 1,...,k 
\\
\end{array}
$$

# Probabilistic perspective

So far, we have considered sparse coding in the context of finding a sparse, over-complete set of basis vectors to span our input space. Alternatively, we may also approach sparse coding from a probabilistic perspective as a generative model.

## Probabilistic models

Consider the problem of modelling natural images as the linear superposition of $k$ independent source features $\phi_i$ with some additive noise $\nu$:
$$
\begin{align}
\mathbf{x} = \sum_{i=1}^k a_i \mathbf{\phi}_{i} + \nu(\mathbf{x})
\end{align}
$$
Assuming $\nu$ is Gaussian white noise with variance $\sigma^2$, we have probabilistic model:
$$
\begin{align}
P(\mathbf{x} \mid \mathbf{a}, \mathbf{\phi}) = \frac{1}{Z} \exp\left(- \frac{(\mathbf{x}-\sum^{k}_{i=1} a_i \mathbf{\phi}_{i})^2}{2\sigma^2}\right)
\end{align}
$$
We also specify the prior distribution $P(\mathbf{a})$. Assuming the independence of our source features, we can factorize our prior probability as:
$$
\begin{align}
P(\mathbf{a}) = \prod_{i=1}^{k} P(a_i)
\end{align}
$$
At this point, we would like to incorporate our sparsity assumption – the assumption that any single image is likely to be the product of relatively few source features. Therefore, we would like the probability distribution of $a_i$ to be peaked at zero and have high kurtosis. A convenient parameterization of the prior distribution is:
$$
\begin{align}
P(a_i) = \frac{1}{Z}\exp(-\beta S(a_i))
\end{align}
$$
where $S(a_i)$ is a function determining the shape of the prior distribution.

We can now give the probabilistic model for data $x$:
$$
\begin{align}
P(\mathbf{x} \mid \mathbf{\phi}) = \int P(\mathbf{x} \mid \mathbf{a}, \mathbf{\phi}) P(\mathbf{a}) d\mathbf{a}
\end{align}
$$

## Optimization Objective

Our goal is to find a set of basis feature vectors $\phi$ such that the distribution of images $P(\mathbf{x}|\phi)$ is as close as possible to the empirical distribution of our input data $P^*(\mathbf{x})$.  One method of doing so is to minimize the KL divergence between those two distributions:
$$
\begin{align}
\text{argmin}_{\phi}{KL}(P^*(\mathbf{x})||P(\mathbf{x}\mid\mathbf{\phi})) = \int P^*(\mathbf{x}) \log \left(\frac{P^*(\mathbf{x})}{P(\mathbf{x}\mid\mathbf{\phi})}\right)d\mathbf{x}
\end{align}
$$
Since the empirical distribution $P^*(\mathbf{x})$ is constant across our choice of $\phi$, this is equivalent to maximizing the log-likelihood of $P(\mathbf{x}|\phi)$:
$$
\begin{align}
\mathbf{\phi}^*=\text{argmax}_{\mathbf{\phi}} E\left[ \log(P(\mathbf{x} \mid \mathbf{\phi})) \right]
\end{align}
$$

# Learning

Learning a set of basis vectors $\phi$ using sparse coding consists of performing two separate optimizations, the first being an optimization over coefficients $a_i$ for each training example $x$ and the second an optimization over basis vectors $\phi$ across many training examples at once.

As described above, a significant limitation of sparse coding is that even after a set of basis vectors have been learnt, in order to “encode” a new data example, optimization must be performed to obtain the required coefficients. This significant “runtime” cost means that sparse coding is computationally expensive to implement even at test time especially compared to typical feedforward architectures.

 

