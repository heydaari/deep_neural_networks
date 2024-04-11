# Deep Neural Networks
## Deep Fully Connected Feed Forward Neural Networks with multiple approaches !

![alt text](https://th.bing.com/th/id/R.bfd6e48ea1655ed2ebbb7db3ab1ef13a?rik=VMK9JfyHcKyQ2g&pid=ImgRaw&r=0)


## Introduction

This repository contains implementations of deep neural networks using three different approaches: JAX, TensorFlow Sequential, and TensorFlow Functional. Each approach is encapsulated in its own Python script, providing a clear and concise demonstration of each method.

## Contents

1. `nn_from_scratch_JAX.py`: This script demonstrates how to implement a deep neural network using the JAX library. JAX is a numerical computing library that combines NumPy, automatic differentiation, and GPU/TPU acceleration. The script includes functions for initializing network parameters, defining the ReLU and softmax activation functions, predicting output, calculating accuracy, and updating parameters. It also includes code for loading the MNIST dataset and training the model.

2. `tensorflow_sequential.py`: This script shows the implementation of a deep neural network using the Sequential API of TensorFlow. The Sequential API allows you to create models layer-by-layer in a step-by-step fashion. The script includes code for loading the MNIST dataset, preprocessing the data, defining the model architecture, compiling the model, and training the model.

3. `tensorflow_functional.py`: This script contains an implementation of a deep neural network using the Functional API of TensorFlow. The Functional API is a way to create models that are more flexible than the Sequential API. It can handle models with non-linear topology, shared layers, and even multiple inputs or outputs. The script includes code for loading the MNIST dataset, preprocessing the data, defining the model architecture using the Functional API, compiling the model, and training the model.

