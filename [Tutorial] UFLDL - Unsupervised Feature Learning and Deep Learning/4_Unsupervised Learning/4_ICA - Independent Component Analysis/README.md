# Introduction

In **sparse coding**, we wanted to learn an **over-complete basis** for the data. In particular, this implies that the basis vectors that we learn in sparse coding will not be linearly independent. 

While this may be desirable in certain situations, sometimes we want to learn a **linearly independent basis** for the data. In independent component analysis (**ICA**), this is exactly what we want to do. Further, in ICA, we want to learn not just any linearly independent basis, but an **orthonormal basis** for the data. An orthonormal basis is a basis $(\phi_1, \ldots \phi_n)$ such that $\phi_i \cdot \phi_j = 0$ if $i \neq j$ and $1$ if $i=j$.

Formally, given some data $x$, we would like to learn a set of basis vectors which we represent in the columns of a matrix $W$, such that, firstly, our features are **sparse**; and secondly, our basis is an **orthonormal** basis.

Orthonormal basis:
$$
\begin{align}
W = 
\begin{bmatrix} 
| & | & & |  \\
\mathbf{w_1} & \mathbf{w_2} & \cdots & \mathbf{w_n}  \\
| & | & & | 
\end{bmatrix}       
\qquad \text{where} \quad \mathbf{w_i} \in \R^n, \quad \mathbf{w_i}^T \cdot \mathbf{w_j}=1 \quad \text{if and only if} \quad i \neq j
\end{align}
$$
where $\mathbf{w_i}$ is orthonormal basis vector.

Sparse feature:
$$
\mathbf{f} = \mathbf{W} \mathbf{x} = \sum_{i=1}^{i=n} x_i \mathbf{w_i}
$$
where $\mathbf{f}$ is the sparse feature vector. Due to dimensions in raw data $x$ are correlated, there are some redundant, so the dimension of feature space must be smaller than raw data.

Note in sparse coding, we map **features** to **raw data**, in independent component analysis, we do in the opposite direction, map **raw data** to **features**.

# Optimize

## Objective function

We want to add the sparsity penalty on the features in our objective function. since $\mathbf{W}\mathbf{x}$ is precisely the features that represent the raw data. Thus the sparsity penalty term, $L_1$ norm, as follows:
$$
\lVert Wx \rVert_1
$$
In addition, we want to constraint the basis vectors matrix $\mathbf{W}$ as an orthonormal set. Adding in the orthonormality constraint gives us the full optimization problem for independent component analysis:
$$
\begin{array}{rcl}
     {\rm minimize} & \lVert Wx \rVert_1  \\
     {\rm s.t.}     & WW^T = I \\
\end{array}
$$

## Learning

As is usually the case in deep learning, the above objective function has no simple analytic solution, and to make matters worse, the orthonormality constraint makes it slightly more difficult to optimize for the objective using gradient descent - every iteration of gradient descent must be followed by a step that maps the new basis back to the space of orthonormal bases (hence enforcing the constraint).

In practice, optimizing for the objective function while enforcing the orthonormality constraint (as described in the section below) is feasible but slow. Hence, the use of orthonormal ICA is limited to situations where it is important to obtain an orthonormal basis.

Observe that the constraint $WW^T = I$ implies two other constraints:

- Firstly, since we are learning an orthonormal basis, the number of basis vectors we learn must be less than the dimension of the input. In particular, this means that we cannot learn over-complete bases.
- Secondly, the data must be **ZCA whitened** with no regularization.

Hence, before we even begin to optimize for the orthonormal ICA objective, we must ensure that our data has been **whitened**, and that we are learning an **under-complete** basis.



