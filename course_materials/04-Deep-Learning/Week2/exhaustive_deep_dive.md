
# Month 4, Week 2: Advanced Computer Vision - Exhaustive Deep Dive

## Transfer Learning

### Theoretical Explanation

Transfer learning is a machine learning technique where a model developed for a task is reused as the starting point for a model on a second task. It is a popular approach in deep learning because it allows us to build accurate models in a timesaving way.

**Why does it work?**

The features learned by a model on a large and general dataset (like ImageNet) can be beneficial for a new, specific task. For example, a model trained on ImageNet has learned to recognize generic features like edges, corners, and textures in the early layers, and more complex features like eyes, wheels, and trees in the later layers. These learned features can be very useful for a new task, such as classifying different types of cars.

**Strategies:**

1.  **Feature Extraction:** We treat the pre-trained model as a fixed feature extractor. We remove the final fully-connected layers (the "head") of the model and replace them with a new set of fully-connected layers that are trained on our new dataset. The weights of the convolutional base are frozen.

2.  **Fine-Tuning:** We unfreeze a few of the top layers of a frozen model base and jointly train both the newly-added part of the model and these top layers. This is called "fine-tuning" because we are slightly adjusting the more abstract representations of the model being reused, in order to make them more relevant for the problem at hand.

### Code Snippets

**TensorFlow/Keras:**

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

# Load the VGG16 model with pre-trained ImageNet weights, excluding the top fully-connected layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional base
base_model.trainable = False

# Add a new classification head
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

**PyTorch:**

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Load a pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Freeze all the parameters in the model
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully-connected layer with a new one for our task
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
```

## Famous Architectures

### Inception

The Inception architecture, developed by Google, is designed to perform well even under strict constraints on memory and computational budget. The main innovation of the Inception architecture is the "Inception module".

**Inception Module:**

An Inception module is a block that allows the network to learn spatial features at different scales simultaneously. It consists of multiple parallel convolutional branches with different filter sizes (e.g., 1x1, 3x3, 5x5) and a max-pooling branch. The outputs of these branches are then concatenated and fed into the next layer.

### MobileNet

MobileNets are a class of efficient models for mobile and embedded vision applications. The core idea behind MobileNets is the use of "depthwise separable convolutions".

**Depthwise Separable Convolutions:**

A depthwise separable convolution is a factorized convolution that is composed of two parts:

1.  **Depthwise Convolution:** A single convolutional filter is applied to each input channel.
2.  **Pointwise Convolution:** A 1x1 convolution is used to combine the outputs of the depthwise convolution.

This factorization significantly reduces the number of parameters and computations compared to a standard convolution.

## Object Detection

### R-CNN (Region-based Convolutional Neural Networks)

R-CNN is a two-stage object detection algorithm.

1.  **Region Proposal:** It first generates a set of candidate regions (bounding boxes) in the image that might contain an object. This is typically done using a technique like Selective Search.
2.  **Feature Extraction and Classification:** For each proposed region, it warps the image region to a fixed size and feeds it into a CNN to extract features. Finally, it uses a linear SVM to classify the object in the region.

### YOLO (You Only Look Once)

YOLO is a one-stage object detection algorithm. It is known for its high speed.

**Unified Detection:**

YOLO divides the input image into a grid. For each grid cell, it predicts:

*   **Bounding boxes:** The coordinates and dimensions of the bounding boxes.
*   **Confidence scores:** The probability that the bounding box contains an object.
*   **Class probabilities:** The probability that the object belongs to a particular class.

YOLO performs all these predictions in a single pass, which makes it very fast.

## Image Segmentation

### U-Net

U-Net is a popular architecture for semantic image segmentation. It was originally developed for biomedical image segmentation.

**Encoder-Decoder Architecture:**

U-Net has a U-shaped architecture consisting of two paths:

1.  **Encoder (Contracting Path):** A series of convolutional and max-pooling layers that downsample the input image and extract features at different scales.
2.  **Decoder (Expanding Path):** A series of up-convolutional and convolutional layers that upsample the feature maps and combine them with features from the encoder path through "skip connections".

**Skip Connections:**

The skip connections are the key innovation of U-Net. They allow the decoder to use features from the encoder at different scales, which helps to recover the spatial information lost during downsampling and produce more accurate segmentation maps.
