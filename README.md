# Local RAG Demo

A minimal, local Retrieval-Augmented Generation (RAG) implementation using LangChain, FAISS, and Ollama. 

This project demonstrates how to ingest a local JSON knowledge base, create embeddings, and query them using a local Large Language Model without needing cloud API keys.

## Tech Stack
- **LangChain**: LLM orchestration framework
- **FAISS**: In-memory vector database for similarity search
- **Ollama**: Local model execution
- **Python**: Core logic

## Setup

1. Clone the repository and navigate to the project directory:
   ```bash
   cd rag-demo
   ```
2. Set up a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install langchain langchain-community langchain-ollama faiss-cpu
   ```

4. Ensure [Ollama](https://ollama.ai/) is installed and running on your machine with your preferred model.

## Usage
- **Ingest**: Process `data/sample.json` to generate and store vector embeddings.
- **Query**: Ask questions against the local knowledge base using the RAG pipeline.
