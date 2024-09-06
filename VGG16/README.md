# VGG-16 on CIFAR-100

This repository contains an implementation of the VGG-16 architecture trained on CIFAR-100.

![1_NNifzsJ7tD2kAfBXt3AzEg](https://github.com/user-attachments/assets/f310b6da-e0d1-49f7-bd92-3bfd7f46c584)

Actual Paper Link: [arxiv](https://arxiv.org/abs/1409.1556)

## Model Architecture

- **5 Convolutional Blocks**: Each block consists of Conv2D layers with ReLU activation, followed by MaxPooling.
- **3 Fully Connected Layers**: These layers process the extracted features for classification into 100 classes.

## Training Details

- **Dataset**: CIFAR-100
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 10
