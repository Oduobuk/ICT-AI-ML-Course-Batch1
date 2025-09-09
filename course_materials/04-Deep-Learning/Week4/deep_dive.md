
# Month 4, Week 4: NLP with Deep Learning - Deep Dive

## Text Preprocessing and Representation

*   **Tokenization:** The first step in any NLP pipeline. It's the process of breaking down text into individual units (tokens).
    *   **Word Tokenization:** The most common approach. `"Hello, world!"` becomes `["Hello", ",", "world", "!"]`.
    *   **Subword Tokenization (e.g., BPE, WordPiece):** A more advanced technique that breaks down rare words into smaller, more manageable units. For example, `"unhappiness"` might become `["un", "##happ", "##iness"]`. This helps the model handle words it hasn't seen before.

*   **Embeddings:** Representing words as dense vectors. This is a huge leap from sparse one-hot encodings.
    *   **Why are they better?** Embeddings capture semantic relationships between words. For example, the vectors for `"king"` and `"queen"` will be closer to each other than the vectors for `"king"` and `"apple"`.
    *   **Pre-trained Embeddings (Word2Vec, GloVe):** You can download pre-trained embeddings that have been learned from massive text corpora. This is a form of transfer learning for NLP.

## Applying LSTMs for Text Classification and Sentiment Analysis

*   **The Architecture:** A common and effective architecture for text classification is:
    1.  **Embedding Layer:** This layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index. These vectors are learned along with the model if you don't use pre-trained embeddings.
    2.  **LSTM/GRU Layer:** This layer processes the sequence of embeddings and learns to capture the contextual information in the text.
    3.  **Dense Layer:** A final classifier that takes the output of the LSTM/GRU layer and makes a prediction (e.g., positive/negative sentiment).

*   **Example: Sentiment Analysis:** Given a movie review, the model learns to predict whether the review is positive or negative. The LSTM reads the review word by word and builds a representation of the meaning of the text, which is then used by the Dense layer to make the final classification.

## Practical Application: Hugging Face Transformers

*   **The Transformer Revolution:** The Transformer architecture, introduced in 2017, has revolutionized the field of NLP. It relies on a mechanism called "self-attention" to process all words in the input sequence simultaneously, allowing it to capture long-range dependencies more effectively than RNNs.

*   **Hugging Face:** A company and an open-source community that provides a library of pre-trained Transformer models. This has made state-of-the-art NLP accessible to everyone.

*   **Pre-trained Models:**
    *   **BERT (Bidirectional Encoder Representations from Transformers):** A powerful model that is pre-trained on a massive amount of text data. It can be fine-tuned for a wide range of tasks, including text classification, question answering, and named entity recognition.
    *   **GPT (Generative Pre-trained Transformer):** A family of models that are pre-trained to generate text. They can be used for tasks like text completion, summarization, and translation.
    *   **T5 (Text-to-Text Transfer Transformer):** A versatile model that frames every NLP task as a text-to-text problem. For example, to classify a sentence, you would input `"classify: This is a great movie"` and the model would output `"positive"`.

*   **The `pipeline` function:** The easiest way to get started with Hugging Face Transformers. It provides a simple interface for using pre-trained models for a variety of tasks.
