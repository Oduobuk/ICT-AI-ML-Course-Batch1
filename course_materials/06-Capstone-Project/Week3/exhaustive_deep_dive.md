
# Month 6, Week 3: Model Development and Iteration - Exhaustive Deep Dive

## Ensemble Methods

### Theoretical Explanation

Ensemble methods are a class of machine learning algorithms that combine multiple models to improve the performance of the overall model.

*   **Bagging:** An ensemble method that trains multiple models on different bootstrap samples of the training data and then averages their predictions.
*   **Boosting:** An ensemble method that sequentially trains a series of weak learners. Each weak learner is trained to correct the errors of the previous weak learners.
*   **Stacking:** An ensemble method that trains a meta-model to combine the predictions of multiple base models.

## Transfer Learning

### Theoretical Explanation

Transfer learning is a machine learning technique where a model that has been trained on a large dataset is used as a starting point for a model on a new, smaller dataset.

*   **Feature Extraction:** Using a pre-trained model as a fixed feature extractor.
*   **Fine-Tuning:** Unfreezing a few of the top layers of a pre-trained model and jointly training both the newly-added part of the model and these top layers.

## Model Compression

### Theoretical Explanation

Model compression is the process of reducing the size of a machine learning model without significantly affecting its performance.

*   **Pruning:** A technique for removing unnecessary connections from a neural network.
*   **Quantization:** A technique for reducing the precision of the weights in a neural network.
*   **Knowledge Distillation:** A technique for training a smaller model to mimic the behavior of a larger model.
