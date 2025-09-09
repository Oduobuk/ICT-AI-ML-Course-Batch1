# Month 4, Week 2: Advanced Computer Vision - Deep Dive

## Transfer Learning Strategies

Delve into different transfer learning strategies:

*   **Feature Extraction:** Use a pre-trained CNN as a fixed feature extractor. This is ideal when your new dataset is small and similar to the original dataset (e.g., ImageNet). You freeze the convolutional base and only train the classifier head.

    *   **Example:** You have a small dataset of 1,000 images of different types of flowers. You can use a VGG16 model pre-trained on ImageNet, freeze the convolutional layers, and train a new classifier on your flower images.

*   **Fine-Tuning:** Unfreeze and retrain some of the top layers of the pre-trained model. This is effective when your new dataset is large and similar to the original dataset. By fine-tuning, you are adapting the learned features to your specific task.

    *   **Example:** You have a large dataset of 100,000 medical images for disease classification. You can fine-tune the top layers of a ResNet model to adapt the learned features to the nuances of medical images.

## Pre-trained Model Architectures

Explore the key innovations and architectural choices in famous CNNs:

*   **VGGNet (e.g., VGG16, VGG19):**
    *   **Innovation:** Simplicity and depth. VGGNet uses a very small and uniform 3x3 convolutional filter, which allows it to build very deep networks.
    *   **Trade-offs:** VGGNet is very large and computationally expensive.

*   **ResNet (e.g., ResNet50, ResNet101):**
    *   **Innovation:** Residual connections (or skip connections). These connections allow the gradient to flow directly through the network, which helps to mitigate the vanishing gradient problem and enables the training of very deep networks.
    *   **Trade-offs:** ResNet is more complex than VGGNet, but it is more accurate and has fewer parameters.

*   **Inception (e.g., GoogLeNet, InceptionV3):**
    *   **Innovation:** The Inception module. This module uses parallel convolutional filters of different sizes (1x1, 3x3, 5x5) to capture features at multiple scales.
    *   **Trade-offs:** The Inception architecture is more complex than ResNet, but it is more efficient and has fewer parameters.

*   **MobileNet (e.g., MobileNetV1, MobileNetV2):**
    *   **Innovation:** Depthwise separable convolutions. This type of convolution significantly reduces the number of parameters and computations, making it ideal for mobile and embedded devices.
    *   **Trade-offs:** MobileNet is less accurate than larger models like ResNet and Inception, but it is much more efficient.

## Object Detection Fundamentals

Deepen the understanding of object detection as a task that involves both localization (predicting the bounding box of an object) and classification (predicting the class of the object).

*   **Challenges:** One of the main challenges in object detection is dealing with objects at different scales and aspect ratios. Another challenge is handling overlapping objects.

## Two-Stage vs. One-Stage Detectors

Introduce the fundamental difference between two-stage and one-stage detectors:

*   **Two-Stage Detectors (e.g., R-CNN, Fast R-CNN, Faster R-CNN):**
    1.  **Region Proposal:** Propose a set of candidate regions where objects might be located.
    2.  **Classification and Refinement:** Classify the objects in the proposed regions and refine the bounding boxes.
    *   **Advantages:** Higher accuracy.
    *   **Disadvantages:** Slower.

*   **One-Stage Detectors (e.g., YOLO, SSD):**
    *   **Unified Detection:** Perform both localization and classification in a single pass.
    *   **Advantages:** Faster.
    *   **Disadvantages:** Lower accuracy compared to two-stage detectors.

## Image Segmentation Concepts

Differentiate between semantic segmentation and instance segmentation:

*   **Semantic Segmentation:** Assign a class label to each pixel in the image. All instances of the same class are labeled with the same color.

    *   **Example:** In an image of a street, all cars are colored blue, all pedestrians are colored red, and the road is colored gray.

*   **Instance Segmentation:** Assign a unique label to each instance of an object in the image. Each instance of the same class is labeled with a different color.

    *   **Example:** In an image of a street, each car is colored with a different shade of blue, and each pedestrian is colored with a different shade of red.

*   **Encoder-Decoder Architectures (e.g., U-Net, FCN):** These architectures are commonly used for segmentation tasks. The encoder downsamples the input image to extract features, and the decoder upsamples the feature maps to generate the segmentation map.