# Introduction

**Convolution** is a mathematical operation. The convolution operation is a powerful tool. In mathematics, it comes up in diverse contexts, ranging from the study of partial differential equations to probability theory. In part because of its role in PDEs, convolution is very important in the physical sciences. It also has an important role in many applied areas, like computer graphics and signal processing.

For us, convolution will provide a number of benefits. Firstly, it will allow us to create much more efficient implementations of convolutional layers than the naive perspective might suggest. Secondly, it will remove a lot of messiness from our formulation. Finally, convolution will give us a significantly different perspective for reasoning about convolutional layers.

Thus it's so important to well understand convolutions.

# Lessons from a Dropped Ball

Imagine we drop a ball from some height onto the ground, where it only has one dimension of motion. *How likely is it that a ball will go a distance $c$ if you drop it and then drop it again from above the point at which it landed?*

Let's break this down:

- After the first drop, it will land $a$ units away from the starting point with probability $f(a)$, where $f$ is the probability distribution.
- Now after this first drop, we pick the ball up and drop it from another height above the point where it first landed. The probability of the ball rolling $b$ units away from the new starting point is $g(b)$, where $g$ may be a different probability distribution if it’s dropped from a different height.

Thus if $a+b=c$, the ball will land $c$ units away from the starting point. So the probability of this happening is simply $f(a) \cdot g(b)$.

![](ProbConv-fagb.png)

In order to find the *total likelihood* of the ball reaching a total distance of $c$, we can’t consider only one possible way of reaching $c$. Instead, we consider *all the possible ways* of partitioning $c$ into two drops $a$ and $b$ and sum over the *probability of each way*. we can denote the total likelihood as:
$$
\sum_{a+b=c} f(a) \cdot g(b)
$$
Turns out, we’re doing a **convolution**! In particular, the convolution of $f$ and $g$, evaluated at $c$ is defined:
$$
(f\ast g)(c) = \sum_{a+b=c} f(a) \cdot g(b)~~~~
$$
If we substitute $b=c-a$, we get:
$$
(f \ast g)(c) = \sum_{a} f(a) \cdot g(c-a)~~~~
$$
This is the standard definition of convolution. We can think, the first drop, it will land at an intermediate position $a$ with probability $f(a)$. If it lands at $a$, it has probability $g(c−a)$ of landing at a position $c$. And to get the convolution, we consider all intermediate positions.

![](ProbConv-OnePath.png)



