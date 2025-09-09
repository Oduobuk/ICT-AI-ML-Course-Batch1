# Month 5, Week 3: Deep Dive - Advanced Deep Learning Architectures

## Recurrent Neural Networks (RNNs)
- **Concept**: Neural networks designed to process sequential data, where the output from the previous step is fed as input to the current step.
- **Vanishing/Exploding Gradients**: Common problems in training deep RNNs, leading to difficulties in learning long-range dependencies.

## Long Short-Term Memory (LSTM) Networks
- **Solution to Vanishing Gradients**: Introduce 'gates' (input, forget, output) to control the flow of information through the cell state.
- **Architecture**: Cell state, hidden state, and gates.
- **Applications**: Speech recognition, machine translation, sentiment analysis.

## Gated Recurrent Unit (GRU) Networks
- **Simplified LSTM**: Combines the forget and input gates into an 'update gate' and merges the cell state and hidden state.
- **Fewer Parameters**: Faster to train and less prone to overfitting than LSTMs in some cases.

## Convolutional Neural Networks (CNNs) for Sequence Data
- **1D Convolutions**: Applying convolutional filters across a sequence (e.g., text, time series) to extract local features.
- **Dilated Convolutions**: Expanding the receptive field without increasing parameters, useful for capturing longer-range dependencies.

## Transformer Networks
- **Attention Mechanism**: Allows the model to weigh the importance of different parts of the input sequence when processing each element.
- **Self-Attention**: Relates different positions of a single sequence to compute a representation of the sequence.
- **Multi-Head Attention**: Multiple attention mechanisms run in parallel to capture different aspects of relationships.
- **Positional Encoding**: Adds information about the relative or absolute position of tokens in the sequence, as Transformers do not inherently process sequence order.
- **Encoder-Decoder Architecture**: Typically used in sequence-to-sequence tasks like machine translation.

## Key Concepts:
- **Sequence Modeling**: Handling data where the order matters.
- **Long-Range Dependencies**: The ability of a model to connect information from distant parts of a sequence.
- **Parallelization**: Transformers' ability to process sequences in parallel, unlike traditional RNNs.