# Local RAG Demo

A robust, local Retrieval-Augmented Generation (RAG) implementation using LangChain, FAISS, and Ollama. 

This project goes beyond a simple tutorial by implementing advanced retrieval techniques to solve real-world RAG challenges:
- **Chunk Size Tuning**: Fine-tuning how documents are split to optimize context window usage.
- **MMR (Maximal Marginal Relevance)**: Diversifying search results to avoid retrieving redundant chunks.
- **HyDE (Hypothetical Document Embeddings)**: Generating hypothetical answers to improve semantic search relevance.
- **Similarity Score Thresholding**: Filtering out irrelevant retrieved chunks to ensure high-quality context.
- **Debugging and Evaluation**: Handling scenarios with improperly retrieved chunks or low similarity scores.

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
   pip install -r requirements.txt
   ```

4. Ensure [Ollama](https://ollama.ai/) is installed and running on your machine with your preferred model.

## Usage
- **Ingest**: Process `data/sample.json` and generate vector embeddings using proper chunking strategies.
- **Query**: Ask questions against the local knowledge base using the advanced RAG pipeline.
