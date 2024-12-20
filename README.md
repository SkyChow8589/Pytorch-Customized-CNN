# ResNeXt with SE-Block and Channel Shuffle in PyTorch

This repository contains a custom implementation of the ResNeXt architecture with enhancements such as SENet (Squeeze-and-Excitation) blocks and channel shuffling, developed using PyTorch. The network achieves **95.5% classification accuracy on the CIFAR-10 dataset**.

## Features

- **ResNeXt Architecture**: Implements a modular, scalable deep learning model with grouped convolutions for efficient representation learning.
- **Channel Shuffle**: Improves information flow between grouped convolutions, ensuring better feature interactions.
- **SENet Block**: Dynamically recalibrates channel-wise feature responses using a lightweight attention mechanism.
- **Custom Training Pipeline**: Includes optimization using MultiStepLR, advanced data augmentation, and regularization techniques.

## Code Structure
At ./models/mynet.py

- **`channel_shuffle`**: Implements channel shuffling to enhance information exchange across groups in grouped convolutions.
- **`SE_Block`**: Adds channel attention to recalibrate feature maps.
- **`Block`**: Basic building block combining grouped convolutions, SE blocks, and channel shuffle.
- **`ResNeXt`**: Core architecture with scalable cardinality, bottleneck width, and number of layers.
- **Utility Functions**:
  - `ResNeXt50_32x4d()`: Constructs ResNeXt-50 with 32 groups and bottleneck width of 4.
  - `ResNeXt101_32x4d()`: Constructs ResNeXt-101 with similar configurations.
  - `test_resnext()`: Provides a basic test case for the model architecture.

## Training Details

- **Dataset**: CIFAR-10
- **Optimizer**: SGD with momentum
- **Learning Rate Scheduler**: MultiStepLR
- **Loss Function**: CrossEntropyLoss
- **Augmentation**: Standard data augmentation techniques such as random cropping and flipping.
- **Batch Size**: 128
- **Epochs**: 200

## Results

- **Accuracy**: 95.5% on the CIFAR-10 test set.
- **Performance Enhancements**:
  - Channel Shuffle increased feature diversity across grouped convolutions.
  - SENet blocks provided adaptive recalibration of feature responses.
  - MultiStepLR ensured efficient learning rate decay during training.

## Usage

### Prerequisites

Ensure PyTorch is installed:
```bash
pip install torch torchvision

