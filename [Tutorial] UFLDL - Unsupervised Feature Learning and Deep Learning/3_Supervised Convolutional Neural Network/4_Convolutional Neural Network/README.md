# Introduction

A Convolutional Neural Network (**CNN**) is comprised of one or more convolutional layers (often with a subsampling step) and then followed by one or more fully connected layers as in a standard **multilayer neural network**. 

The architecture of a CNN is designed to take advantage of the **2D structure** of data  (such as an image or a speech signal). This is achieved with **local connections** and **shared weights** followed by some form of pooling which results in **translation invariant features**. Another benefit of CNNs is that they are easier to train and have many **fewer parameters** than fully connected networks with the same number of hidden units.

# Architecture

A CNN consists of a number of convolutional and subsampling layers optionally followed by fully connected layers.

## Convolutional layers

Assume our CNN is used to image classification, the input to a convolutional layer is $w \times h \times r$ where $w$ is the width and $h$ is the height of the image and $r$ is the number of channels, e.g. an RGB image has $r=3$. 

 

