# S6 Assignment

This repository contains the code for the S6 Assignment, which is a convolutional neural network (CNN) model trained on the MNIST dataset.

### **Dataset details**

- Dataset Size - Total images - 70000
    1. Train images - 60000
    2. Test images - 10000
- Image profile -
    1. Size - 1x28x28 (Channel x Width x Height)
    2. Grayscale - single channel
- Dataset source - https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html


### **Training results**

- Epochs - 20
- Optimizer - SGD (Stochastic gradient descent)
- Learning rate - 0.01 with LR scheduler
- Batch size - 128 (depends on device where this model gets executed)

Adjusting learning rate of group 0 to 1.0000e-03.
loss=0.0043894764967262745 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.63it/s]
Test set: Average loss: 0.0229, Accuracy: 9933/10000 (99%)

Adjusting learning rate of group 0 to 1.0000e-03.
loss=0.04008495807647705 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.85it/s]
Test set: Average loss: 0.0240, Accuracy: 9931/10000 (99%)

Adjusting learning rate of group 0 to 1.0000e-03.
loss=0.012727566994726658 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.37it/s]
Test set: Average loss: 0.0238, Accuracy: 9933/10000 (99%)

Adjusting learning rate of group 0 to 1.0000e-03.
loss=0.007890218868851662 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 26.86it/s]
Test set: Average loss: 0.0237, Accuracy: 9932/10000 (99%)
