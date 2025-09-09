
# Month 4, Week 3: Recurrent Neural Networks (RNNs) - Deep Dive

## Handling Sequence Data

*   **What is Sequence Data?:** Understand that sequence data is any data where the order matters. This includes time series (stock prices, weather), text (sentences, paragraphs), audio, and more.
*   **Why not Standard NNs?:** Realize that feedforward networks (Dense layers) process each input independently. They have no memory of previous inputs, making them unsuitable for tasks where context is crucial.
*   **The RNN Solution:** RNNs introduce the concept of a hidden state that is passed from one timestep to the next. This hidden state acts as a memory, allowing the network to retain information about past elements in the sequence.

## The Vanishing/Exploding Gradient Problem

*   **The Core Issue:** This is a major obstacle in training deep networks, especially RNNs on long sequences. It stems from the repeated multiplication of gradients during backpropagation.
*   **Vanishing Gradients:** Imagine a long chain of multiplications of numbers less than 1. The result gets exponentially smaller, approaching zero. This means the early layers of the network learn very slowly or not at all, as the error signal from the output can't reach them. The RNN fails to learn long-range dependencies.
    *   **Example:** In a long paragraph, the model might forget the subject of the sentence by the time it reaches the end.
*   **Exploding Gradients:** Now imagine multiplying numbers greater than 1. The result grows exponentially, leading to massive updates to the weights. This makes the training process unstable, and the model's weights can become NaN (Not a Number).
    *   **Example:** The model's predictions can swing wildly from one training step to the next.
*   **Solutions:**
    *   **Gradient Clipping:** A simple but effective technique. If the norm of the gradient exceeds a certain threshold, it is scaled down.
    *   **Better Architectures:** The most effective solution is to use more sophisticated RNN variants like LSTMs and GRUs.

## Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) Networks

*   **The Gating Mechanism:** LSTMs and GRUs introduce "gates" – neural network layers that regulate the flow of information. These gates learn which information is important to keep and which to discard.

*   **LSTM (Long Short-Term Memory):**
    *   **Cell State:** The key to the LSTM is the cell state, a horizontal line running down the entire chain. It's like a conveyor belt, allowing information to flow along it with only minor linear interactions. This is what helps preserve the gradient over long sequences.
    *   **Gates:**
        1.  **Forget Gate:** Decides what information to throw away from the cell state. (e.g., Forget the old subject when a new one is introduced).
        2.  **Input Gate:** Decides which new information to store in the cell state. (e.g., Store the new subject).
        3.  **Output Gate:** Decides what to output from the cell state. (e.g., Output the current subject to influence the prediction of the verb).

*   **GRU (Gated Recurrent Unit):**
    *   **A Simpler Alternative:** The GRU is a simplified version of the LSTM with fewer parameters.
    *   **Gates:**
        1.  **Update Gate:** Combines the function of the LSTM's forget and input gates. It decides how much of the past information to keep and how much new information to add.
        2.  **Reset Gate:** Decides how much of the past information to forget.
    *   **When to use which?:** There's no hard rule. LSTMs are a good default choice, but GRUs are worth trying as they are computationally cheaper and may perform better on smaller datasets.
