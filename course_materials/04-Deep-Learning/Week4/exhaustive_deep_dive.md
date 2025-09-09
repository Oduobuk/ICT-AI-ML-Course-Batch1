
# Month 4, Week 4: NLP with Deep Learning - Exhaustive Deep Dive

## Text Preprocessing and Representation

### Theoretical Explanation

Before we can feed text data into a deep learning model, we need to preprocess it and convert it into a numerical representation.

**Tokenization:**

Tokenization is the process of breaking down a piece of text into smaller units, called tokens. These tokens can be words, subwords, or characters.

*   **Word Tokenization:** Splitting a sentence into words.
*   **Subword Tokenization:** Splitting a word into smaller units. This is useful for handling rare words and out-of-vocabulary words.
*   **Character Tokenization:** Splitting a sentence into characters.

**Embeddings:**

Embeddings are a way of representing words as dense vectors in a low-dimensional space. This is in contrast to one-hot encoding, which represents words as sparse vectors in a high-dimensional space.

*   **Word2Vec:** A popular algorithm for learning word embeddings from a large corpus of text.
*   **GloVe (Global Vectors for Word Representation):** Another popular algorithm for learning word embeddings.
*   **FastText:** An extension of Word2Vec that can also learn embeddings for subwords.

### Code Snippets

**TensorFlow/Keras:**

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding
data = pad_sequences(sequences, maxlen=100)
```

**PyTorch:**

```python
import torch
from torchtext.data import get_tokenizer

# Tokenization
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer("Hello, world!")
```

## Applying LSTMs for Text Classification and Sentiment Analysis

### Theoretical Explanation

LSTMs are well-suited for text classification and sentiment analysis tasks because they can learn long-range dependencies in text.

**Architecture:**

A typical architecture for text classification with LSTMs consists of three layers:

1.  **Embedding Layer:** Converts the input text into a sequence of word embeddings.
2.  **LSTM Layer:** Processes the sequence of word embeddings and learns to extract features from the text.
3.  **Dense Layer:** Classifies the text based on the features extracted by the LSTM layer.

### Code Snippets

**TensorFlow/Keras:**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(10000, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
```

## Hugging Face Transformers

### Theoretical Explanation

Hugging Face Transformers is a popular library that provides pre-trained models for a wide range of NLP tasks.

**Transformer Architecture:**

The Transformer architecture is a new type of neural network architecture that was introduced in the paper "Attention Is All You Need". It is based on the concept of self-attention, which allows the model to weigh the importance of different words in the input text.

**Pre-trained Models:**

Hugging Face provides a wide range of pre-trained models, including:

*   **BERT (Bidirectional Encoder Representations from Transformers):** A powerful pre-trained model that can be fine-tuned for a variety of NLP tasks.
*   **GPT-2 (Generative Pre-trained Transformer 2):** A large-scale language model that can generate human-like text.
*   **T5 (Text-to-Text Transfer Transformer):** A model that can be used for a variety of NLP tasks by framing them as text-to-text problems.

### Code Snippets

**Hugging Face Transformers:**

```python
from transformers import pipeline

# Sentiment analysis
classifier = pipeline('sentiment-analysis')
result = classifier('I love this movie!')

# Text generation
generator = pipeline('text-generation', model='gpt2')
result = generator('Hello, I am a language model', max_length=30, num_return_sequences=5)
```
