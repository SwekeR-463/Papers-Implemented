# MicroVGG


This repository contains my implementation of a micro VGG, a tiny version of VGG, designed for classifying images of dogs and cats from the CIFAR-10 dataset.


![Screenshot 2024-08-25 091014](https://github.com/user-attachments/assets/7f731575-b2bb-4e4e-a578-bde0290926f0)


Actual Paper Link: [arxiv](https://arxiv.org/abs/1409.1556)

<h1>Architecture Overview</h1>
The micro VGG model is a simplified version of the original VGG architecture:
<ul>
<li>Conv1: 3x3, 32 filters</li>
<li>Conv2: 3x3, 64 filters</li>
<li>Conv3: 3x3, 128 filters</li>
<li>FC1: 512 units</li>
<li>FC2: 2 units (Dog, Cat)</li>
</ul>

The micro VGG-like model performs well on the dog vs. cat classification task, demonstrating the power of even modest convolutional architectures getting <b>90%+</b> accuracy on every training loop.
