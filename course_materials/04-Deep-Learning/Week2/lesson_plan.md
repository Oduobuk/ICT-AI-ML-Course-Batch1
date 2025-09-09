# Month 4, Week 2: Advanced Computer Vision

## Lesson Plan

*   **Transfer Learning:**
    *   **Concept:** Learn to leverage knowledge from models pre-trained on large datasets (like ImageNet) for new, specific tasks.
    *   **Strategies:** Understand the difference between feature extraction and fine-tuning, and when to use each.
    *   **Hands-on:** Apply transfer learning using a pre-trained model (e.g., VGG16 or ResNet) to a new image classification dataset. Compare the performance with a CNN trained from scratch.

*   **Famous CNN Architectures:**
    *   **Conceptual Overview:** Understand the design principles and innovations behind widely used CNN models like VGG, ResNet, Inception, and MobileNet.
    *   **Discussion:** Analyze the trade-offs of each architecture in terms of accuracy, speed, and model size.

*   **Object Detection:**
    *   **Introduction:** Get an overview of object detection as a computer vision task that involves both localization and classification.
    *   **Algorithms:** Learn about the two main families of object detection algorithms: two-stage detectors (e.g., R-CNN family) and one-stage detectors (e.g., YOLO, SSD).
    *   **Hands-on:** Implement a simple object detection pipeline using a pre-trained YOLO model.

*   **Image Segmentation:**
    *   **Introduction:** Understand the difference between semantic segmentation and instance segmentation.
    *   **Architectures:** Learn about encoder-decoder architectures like U-Net for segmentation tasks.

## Reading Assignments

*   **Primary Reading:**
    *   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3rd Edition" by Aurélien Géron:**
        *   Chapter 14: Deep Computer Vision Using Convolutional Neural Networks (focus on Transfer Learning and famous architectures).
    *   **"Deep Learning with Python, 2nd Edition" by François Chollet:**
        *   Chapter 8: Introduction to deep learning for computer vision (focus on Transfer Learning).

*   **Further Reading:**
    *   **"Going Deeper with Convolutions" (Inception paper):** [https://arxiv.org/abs/1409.4842](https://arxiv.org/abs/1409.4842)
    *   **"Deep Residual Learning for Image Recognition" (ResNet paper):** [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
    *   **"You Only Look Once: Unified, Real-Time Object Detection" (YOLO paper):** [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)
    *   **"U-Net: Convolutional Networks for Biomedical Image Segmentation" (U-Net paper):** [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

## Assignments

*   **Assignment 1: Transfer Learning for Image Classification**
    *   Choose a dataset of your interest (e.g., from Kaggle or TensorFlow Datasets).
    *   Implement two models for image classification:
        1.  A simple CNN trained from scratch.
        2.  A pre-trained model (e.g., VGG16 or ResNet) using transfer learning (feature extraction or fine-tuning).
    *   Compare the performance of the two models and write a brief report on your findings.

*   **Assignment 2: Object Detection with YOLO**
    *   Use a pre-trained YOLO model (e.g., from TensorFlow Hub or PyTorch Hub) to perform object detection on a set of images or a video.
    *   Visualize the detected objects and their bounding boxes.
    *   (Optional) Try to fine-tune the YOLO model on a custom dataset.

*   **Assignment 3: Research and Presentation**
    *   Choose one of the following topics:
        *   The evolution of the R-CNN family (from R-CNN to Faster R-CNN).
        *   The architecture of the SSD (Single Shot MultiBox Detector) model.
        *   The architecture of the FCN (Fully Convolutional Network) for semantic segmentation.
    *   Prepare a short presentation (5-10 minutes) explaining the key ideas and innovations of your chosen topic.