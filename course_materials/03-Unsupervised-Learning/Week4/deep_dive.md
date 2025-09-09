# Month 3, Week 4: Introduction to Neural Networks - Deep Dive

## Perceptron Learning Algorithm

*   **Perceptron:** The simplest type of neural network. It consists of a single neuron that takes a set of binary inputs and produces a binary output.
*   **Learning Rule:** The perceptron learning rule is a simple algorithm for updating the weights of a perceptron. It works by adjusting the weights in the direction that reduces the error.
*   **Limitations:** The perceptron can only learn linearly separable patterns. It cannot learn non-linearly separable patterns, such as the XOR problem.

## Multilayer Perceptrons (MLPs)

*   **MLP:** A type of neural network that consists of multiple layers of perceptrons. MLPs can learn complex, non-linear relationships between the inputs and the outputs.
*   **Universal Approximation Theorem:** A theorem that states that an MLP with a single hidden layer can approximate any continuous function to any desired degree of accuracy.

## Activation Functions

*   **Activation Function:** A function that is applied to the output of a neuron to introduce non-linearity into the network.
*   **Common Activation Functions:**
    *   **Sigmoid:** A function that maps any real value to a value between 0 and 1.
    *   **ReLU (Rectified Linear Unit):** A function that outputs the input if it is positive and 0 otherwise.
    *   **Leaky ReLU:** A variation of ReLU that outputs a small positive value if the input is negative.
    *   **ELU (Exponential Linear Unit):** A variation of ReLU that is similar to Leaky ReLU, but it uses an exponential function instead of a linear function.
    *   **Tanh (Hyperbolic Tangent):** A function that maps any real value to a value between -1 and 1.
    *   **Softmax:** A function that is used for multi-class classification. It takes a vector of scores as input and outputs a vector of probabilities that sum to 1.

## Backpropagation Algorithm

*   **Backpropagation:** An algorithm for efficiently computing the gradients of the loss function with respect to the weights of the network. It works by traversing the computational graph backwards and applying the chain rule.

## Loss Functions for Neural Networks

*   **Loss Function:** A function that measures the difference between the predicted values and the actual values.
*   **Common Loss Functions:**
    *   **Mean Squared Error (MSE):** A loss function that is used for regression problems.
    *   **Binary Cross-Entropy:** A loss function that is used for binary classification problems.
    *   **Categorical Cross-Entropy:** A loss function that is used for multi-class classification problems.