# ResNet

Implementation of [ResNet](https://arxiv.org/abs/1512.03385) architecture by making a ResNet69 Model. Trained on the Oxford IIIT [Dataset](https://paperswithcode.com/dataset/oxford-iiit-pets).

### How ResNet69 ?
The ResNet-69 is structured by stacking these BasicBlocks (which is the implementation of residual connections) in the following configuration:
* Layer 1: 3 blocks
* Layer 2: 4 blocks
* Layer 3: 12 blocks
* Layer 4: 3 blocks

This results in a total of 69 layers (3 layers per block across the 22 blocks in total).

### Outputs 
