
# Month 4, Week 4: Natural Language Processing (NLP) with Deep Learning

## Lesson Plan

*   **Text Preprocessing and Representation:**
    *   **Concept:** Learn how to prepare text data for deep learning models, including tokenization and word embeddings.
    *   **Hands-on:** Use TensorFlow/Keras or PyTorch to tokenize text and create word embeddings.

*   **Applying LSTMs for Text Classification:**
    *   **Concept:** Understand how to use LSTMs for text classification tasks like sentiment analysis.
    *   **Hands-on:** Build and train an LSTM model for sentiment analysis on the IMDB dataset.

*   **Hugging Face Transformers:**
    *   **Concept:** Learn about the Transformer architecture and how to use pre-trained models from the Hugging Face library.
    *   **Hands-on:** Use the Hugging Face `pipeline` function to perform sentiment analysis, text generation, and other NLP tasks.

## Reading Assignments

*   **Primary Reading:**
    *   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3rd Edition" by Aurélien Géron:**
        *   Chapter 16: Natural Language Processing with RNNs and Attention.
    *   **"Deep Learning with Python, 2nd Edition" by François Chollet:**
        *   Chapter 11: Deep learning for text.
    *   **"Natural Language Processing with Transformers" by Lewis Tunstall, Leandro von Werra, and Thomas Wolf:**
        *   Chapter 1: Introduction to Transformers.
        *   Chapter 2: Text Classification with Transformers.

*   **Further Reading:**
    *   **"Attention Is All You Need" (Transformer paper):** [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
    *   **"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (BERT paper):** [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

## Assignments

*   **Assignment 1: Text Classification with LSTMs and Transformers**
    *   Choose a text classification dataset (e.g., from Kaggle or Hugging Face Datasets).
    *   Build and train two models for text classification:
        1.  An LSTM model.
        2.  A pre-trained Transformer model from Hugging Face (e.g., BERT or DistilBERT).
    *   Compare the performance of the two models and write a brief report on your findings.

*   **Assignment 2: Text Generation with GPT-2**
    *   Use the Hugging Face `pipeline` function to generate text with the GPT-2 model.
    *   Experiment with different prompts and parameters to generate different styles of text.
    *   (Optional) Fine-tune the GPT-2 model on a dataset of your choice to generate text in a specific style.

*   **Assignment 3: Exploring the Hugging Face Hub**
    *   Explore the Hugging Face Hub ([https://huggingface.co/models](https://huggingface.co/models)).
    *   Find a pre-trained model for a task that interests you (e.g., question answering, summarization, translation).
    *   Use the `pipeline` function to try out the model on a few examples.
    *   Write a short summary of the model and its capabilities.
