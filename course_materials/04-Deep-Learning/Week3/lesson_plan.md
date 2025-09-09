
# Month 4, Week 3: Recurrent Neural Networks (RNNs) for Sequence Data

## Lesson Plan

*   **Handling Sequence Data:**
    *   **Concept:** Understand the nature of sequence data and why standard feedforward networks are not suitable.
    *   **Hands-on:** Prepare text data for an RNN, including tokenization and creating sequences.

*   **Recurrent Neural Networks (RNNs):**
    *   **Architecture:** Learn the basic architecture of an RNN and how the hidden state allows it to have "memory".
    *   **The Vanishing/Exploding Gradient Problem:** Understand why RNNs are difficult to train on long sequences.

*   **Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) Networks:**
    *   **Concept:** Learn how LSTMs and GRUs use gating mechanisms to overcome the vanishing gradient problem and learn long-range dependencies.
    *   **Hands-on:** Build and train an LSTM or GRU model for a sequence-based task, such as sentiment analysis or text generation.

## Reading Assignments

*   **Primary Reading:**
    *   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3rd Edition" by Aurélien Géron:**
        *   Chapter 15: Processing Sequences Using RNNs and CNNs (focus on RNNs, LSTMs, and GRUs).
    *   **"Deep Learning with Python, 2nd Edition" by François Chollet:**
        *   Chapter 6: The universal workflow of machine learning (re-read, focusing on the sequence data aspect).
        *   Chapter 11: Deep learning for text (focus on RNNs, LSTMs, and GRUs).

*   **Further Reading:**
    *   **"Understanding LSTM Networks" by Chris Olah:** [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
    *   **"The Unreasonable Effectiveness of Recurrent Neural Networks" by Andrej Karpathy:** [http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## Assignments

*   **Assignment 1: Sentiment Analysis with RNNs**
    *   Use the IMDB movie review dataset (available in Keras).
    *   Build and train three models for sentiment analysis:
        1.  A simple RNN.
        2.  An LSTM.
        3.  A GRU.
    *   Compare the performance of the three models and write a brief report on your findings.

*   **Assignment 2: Text Generation**
    *   Choose a text dataset (e.g., a book from Project Gutenberg, song lyrics, etc.).
    *   Build and train an LSTM or GRU model to generate text in the style of the chosen dataset.
    *   Experiment with different model architectures and hyperparameters to improve the quality of the generated text.

*   **Assignment 3: Time Series Forecasting**
    *   Choose a time series dataset (e.g., stock prices, weather data).
    *   Build and train an LSTM or GRU model to forecast future values in the time series.
    *   Evaluate the performance of your model using appropriate metrics (e.g., Mean Squared Error, Mean Absolute Error).
