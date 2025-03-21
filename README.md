# The Chatty Python Experiment

This experimental project is a Python programming chatbot built as a **Retrieval-Augmented Generation (RAG) model**. It assists users by delivering context-aware answers and code examples using advanced prompt engineering and efficient retrieval techniques. Try the Chatty Python here:
https://chattypython.streamlit.app/

## Overview

- **Purpose:**  
  This project was developed to experiment with prompt engineering and RAG models, demonstrating how detailed information can be hidden in metadata to speed up retrieval while still providing comprehensive answers.

- **Data Sources:**  
  The chatbot uses data scraped from the following pages:
  - [Python FAQ](https://docs.python.org/3/faq/index.html)  
    Only the questions from the official Python FAQ are embedded, with the detailed answers hidden in the metadata of the vector database.
  - [Sci-kit Learn Auto Examples](https://scikit-learn.org/stable/auto_examples/index.html)  
    Only the tooltips (or summaries) of the examples are embedded, while the full examples (including code snippets) are again stored in the metadata.
  
  Two independent embeddings were created to ensure that the model always delivers both a relevant answer and a corresponding example.

## The RAG Model

The chatbot leverages a RAG approach by combining two separate retrieval systems:
- **FAQ Retriever:** Quickly fetches relevant FAQ questions.
- **Sci-kit Example Retriever:** Retrieves tooltips of examples, with complete examples hidden in the metadata.

This combination ensures that every query is answered with both a contextually relevant explanation and a practical example.

## Modes

The chatbot offers three distinct response modes that highlight different prompt engineering strategies:

- **Neutral Mode:**  
  Provides concise and straightforward answers strictly related to Python programming.

- **Analytic Mode:**  
  Delivers detailed, step-by-step explanations along with native code snippets.

- **Enthusiastic Mode:**  
  Uses energetic and exuberant language to celebrate Python’s capabilities while delivering accurate answers.

These modes allow users to explore different communication styles and see the impact of advanced prompt engineering. Some measures where taken so the Chatbot only answers Python-related questions... However, this is not always working!

## How It Works

1. **Custom Embeddings:**  
   - **FAQ Embeddings:** Only the questions are embedded, while the answers remain hidden in metadata.
   - **Sci-kit Example Embeddings:** Only the example pointers are embedded, with full examples (including code) stored in metadata.

2. **RAG Retrieval Chain:**  
   A custom retrieval chain combines outputs from both embeddings, loading the content hidden in the metadata into the context window of the prompt, ensuring that each query returns both a relevant answer and a useful example, if helpful.

3. **Interactive Interface:**  
   The chatbot prototype runs on Streamlit, offering an interactive chat interface where users can ask questions and receive formatted responses.


