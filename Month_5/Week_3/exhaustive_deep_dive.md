# Month 5, Week 3: Exhaustive Deep Dive - Advanced Deep Learning Architectures

## Recurrent Neural Networks (RNNs) - In-depth

### Mathematical Formulation:
An RNN processes a sequence $x = (x_1, x_2, ..., x_T)$ by iterating through its elements. At each time step $t$, the hidden state $h_t$ is updated based on the current input $x_t$ and the previous hidden state $h_{t-1}$:

$h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$

And the output $y_t$ is computed as:

$y_t = W_{hy}h_t + b_y$

Where:
- $W_{hh}$, $W_{xh}$, $W_{hy}$ are weight matrices.
- $b_h$, $b_y$ are bias vectors.
- $f$ is a non-linear activation function (e.g., tanh or ReLU).

### Challenges:
- **Vanishing Gradients**: During backpropagation through time (BPTT), gradients can shrink exponentially, making it difficult for the network to learn long-term dependencies. This is due to repeated multiplication of the weight matrix $W_{hh}$ and the derivative of the activation function $f'$.
- **Exploding Gradients**: Conversely, gradients can grow exponentially, leading to unstable training. This can often be mitigated by gradient clipping.

## Long Short-Term Memory (LSTM) Networks - Detailed Exploration

### Solving Vanishing Gradients:
LSTMs were introduced to combat the vanishing gradient problem by incorporating a 'cell state' ($C_t$) that runs straight through the entire chain, allowing information to be carried forward unchanged. This is regulated by various 'gates'.

### The Gates:
1.  **Forget Gate ($f_t$)**: Decides what information to throw away from the cell state.
    $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
2.  **Input Gate ($i_t$)**: Decides what new information to store in the cell state.
    $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
    $ ightilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$ (Candidate for cell state)
3.  **Update Cell State**: The new cell state is a combination of the old cell state (forgotten parts) and the candidate cell state (new information).
    $C_t = f_t * C_{t-1} + i_t * ightilde{C}_t$
4.  **Output Gate ($o_t$)**: Decides what part of the cell state to output.
    $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
    $h_t = o_t * \tanh(C_t)$

Where $\sigma$ is the sigmoid activation function, and $*$ denotes element-wise multiplication.

### Advantages:
- Effectively captures long-term dependencies.
- Widely used in sequence modeling tasks.

## Gated Recurrent Unit (GRU) Networks - Simplified Gating

### Architecture:
GRUs are a simpler variant of LSTMs, combining the forget and input gates into a single 'update gate' ($z_t$) and merging the cell state and hidden state. They also have a 'reset gate' ($r_t$).

1.  **Update Gate ($z_t$)**: Controls how much of the previous hidden state to carry forward.
    $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$
2.  **Reset Gate ($r_t$)**: Decides how much of the previous hidden state to forget.
    $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$
3.  **Candidate Hidden State ($ ightilde{h}_t$)**: Computes a new hidden state candidate.
    $ ightilde{h}_t = \tanh(W_h \cdot [r_t * h_{t-1}, x_t] + b_h)$
4.  **Update Hidden State**: The final hidden state is a linear interpolation between the previous hidden state and the candidate hidden state.
    $h_t = (1 - z_t) * h_{t-1} + z_t * ightilde{h}_t$

### Advantages:
- Fewer parameters than LSTMs, leading to faster training and potentially less overfitting on smaller datasets.
- Often performs comparably to LSTMs.

## Convolutional Neural Networks (CNNs) for Sequence Data - Beyond Images

### 1D Convolutions:
Instead of 2D filters for images, 1D convolutions apply filters across a single dimension (e.g., time steps in a time series or words in a sentence). This allows them to capture local patterns (n-grams in text, short-term trends in time series).

### Dilated Convolutions:
- **Concept**: Introduce gaps between the elements of the filter, effectively expanding the receptive field without increasing the number of parameters or losing resolution.
- **Benefits**: Useful for capturing longer-range dependencies in sequences while maintaining computational efficiency.

### Applications:
- **Text Classification**: Identifying sentiment or topic.
- **Time Series Forecasting**: Predicting future values based on past patterns.
- **Speech Processing**: Feature extraction from audio signals.

## Transformer Networks - The Attention Revolution

### Self-Attention Mechanism (Scaled Dot-Product Attention):
- **Query (Q), Key (K), Value (V)**: For each token, these three vectors are derived from its embedding. Q is used to query other tokens, K is used to be queried, and V holds the information to be passed on.
- **Calculation**: $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
    - $d_k$ is the dimension of the key vectors, used for scaling to prevent large dot products from pushing the softmax into regions with tiny gradients.
- **Intuition**: Allows each word in a sentence to weigh the importance of all other words in the same sentence when computing its own representation.

### Multi-Head Attention:
- **Concept**: Instead of performing a single attention function, the queries, keys, and values are linearly projected $h$ times with different, learned linear projections. Then, the attention function is applied in parallel to each of these projected versions.
- **Benefits**: Allows the model to jointly attend to information from different representation subspaces at different positions. This enriches the model's ability to capture diverse relationships.

### Positional Encoding:
- **Necessity**: Since Transformers process sequences in parallel and do not have inherent recurrence or convolution, they need a mechanism to understand the order of tokens.
- **Method**: Positional encodings are added to the input embeddings. These are typically sine and cosine functions of different frequencies, allowing the model to learn relative positions.

### Encoder-Decoder Architecture:
- **Encoder**: Stack of identical layers, each with a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. It processes the input sequence.
- **Decoder**: Stack of identical layers, similar to the encoder but with an additional masked multi-head attention layer to prevent attending to future positions (for auto-regressive generation) and an encoder-decoder attention layer that attends to the output of the encoder stack.

### Advantages of Transformers:
- **Parallelization**: Significantly faster training compared to RNNs due to parallel computation of attention.
- **Long-Range Dependencies**: Excellent at capturing very long-range dependencies due to the direct attention mechanism.
- **State-of-the-Art**: Achieved state-of-the-art results in various NLP tasks (e.g., machine translation, text summarization, question answering).

## Further Reading and References:
- **RNNs**: *"Recurrent Neural Networks Tutorial"* by Denny Britz.
- **LSTMs**: *"Long Short-Term Memory"* by Sepp Hochreiter and Jürgen Schmidhuber (1997).
- **GRUs**: *"Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"* by Junyoung Chung et al. (2014).
- **Transformers**: *"Attention Is All You Need"* by Ashish Vaswani et al. (2017).
- **CNNs for NLP**: *"Convolutional Neural Networks for Sentence Classification"* by Yoon Kim (2014).

This exhaustive deep dive provides a comprehensive understanding of advanced deep learning architectures, their mathematical underpinnings, and their applications in sequence modeling.
