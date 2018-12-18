# Introduction

A Convolutional Neural Network (**CNN**) is comprised of one or more convolutional layers (often with a subsampling step) and then followed by one or more fully connected layers as in a standard **multilayer neural network**. 

The architecture of a CNN is designed to take advantage of the **2D structure** of data  (such as an image or a speech signal). This is achieved with **local connections** and **tied weights** followed by some form of pooling which results in **translation invariant features**. Another benefit of CNNs is that they are easier to train and have many **fewer parameters** than fully connected networks with the same number of hidden units.

# Architecture

A CNN consists of a number of convolutional and subsampling layers optionally followed by fully connected layers. 

Assume our CNN is used to image classification, the input to a convolutional layer is $w \times h \times r$ where $w$ is the width and $h$ is the height of the image and $r$ is the number of channels, e.g. an RGB image has $r=3$. 

## Convolutional layers

The convolutional layer will have $k$ **filters** (or **kernels**) of size $n \times n \times q$ where $n$ is smaller than the dimension of the image and $q$ can either be the same as the number of channels $r$ or smaller and may vary for each kernel. The size of the filters gives rise to the locally connected structure which are each convolved with the image to produce $k$ feature maps of size $(w-n+1) \times (h-n+1)$. 

## Subsampling layers

Each feature map is then subsampled typically with mean or max pooling over $p \times p$ contiguous regions where $p$ ranges between 2 for small images (e.g. MNIST) and is usually not more than 5 for larger inputs.

## Non-linear activation

Either **before or after** the subsampling layer an additive bias and sigmoidal nonlinearity is applied to each feature map.

## Fully connected layers

There may be any number of fully connected layers follow the convolutional layers and subsampling layers. The densely connected layers are identical to the layers in a standard multilayer neural network.

# Back Propagation

