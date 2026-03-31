# rag_graph.py — RAG pipeline built as a LangGraph state machine
import json
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
import os

# Config
VECTORSTORE_PATH = "vectorstore"
DATA_PATH = "data/sample.json"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "phi3:mini"
MAX_RETRIES = 2


# State
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retries: int


# Shared resources
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=0)
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.7}
)


# Node 1: retrieve
def retrieve(state: GraphState) -> GraphState:
    print(f"\n[retrieve] Searching vectorstore...")
    docs = retriever.invoke(state["question"])
    topics = [d.metadata["topic"] for d in docs]
    print(f"[retrieve] Got chunks from: {topics}")
    return {"documents": docs}


# Node 2: grade_documents
def grade_documents(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state["documents"]
    retries = state.get("retries", 0)

    print(f"[grade] Checking {len(documents)} chunks for relevance...")

    relevant = []
    for doc in documents:
        # Simple keyword-based grader
        STOP_WORDS = {
            "is",
            "what",
            "how",
            "the",
            "a",
            "an",
            "it",
            "and",
            "or",
            "in",
            "of",
            "to",
            "for",
            "can",
            "we",
            "do",
            "does",
            "why",
            "are",
            "was",
            "be",
            "this",
            "that",
        }

        q_words = set(question.lower().split()) - STOP_WORDS
        doc_words = set(doc.page_content.lower().split()) - STOP_WORDS
        overlap = q_words & doc_words
        if len(overlap) >= 2:
            relevant.append(doc)
            print(f"  PASS - '{doc.metadata['topic']}' (overlap: {overlap})")
        else:
            print(f"  FAIL — '{doc.metadata['topic']}' (overlap: {overlap})")

    return {"documents": relevant, "retries": retries + 1}


# Conditional edge: decide what to do after grading
def decide_after_grade(state: GraphState) -> str:
    if len(state["documents"]) > 0:
        print(f"[decide] Relevant chunks found — moving to generate")
        return "generate"
    if state["retries"] >= MAX_RETRIES:
        print(f"[decide] Max retries hit — generating with no context")
        return "generate"
    print(f"[decide] No relevant chunks — retrying retrieval")
    return "retrieve"


# Node 3: generate
def generate(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state["documents"]

    print(f"[generate] Calling {LLM_MODEL}...")

    if not documents:
        return {"generation": "I don't know based on the provided data."}
    else:
        context = "\n\n".join(doc.page_content for doc in documents)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided data."

Context:
{context}

Question: {question}

Answer:""",
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
    return {"generation": response}


# Build the graph
def build_graph():
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)

    # Add edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grade,
        {"generate": "generate", "retrieve": "retrieve"},
    )
    graph.add_edge("generate", END)

    return graph.compile()


# Run
def ask(app, question: str):
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    result = app.invoke(
        {"question": question, "documents": [], "generation": "", "retries": 0}
    )
    print(f"\nA: {result['generation']}")


def main():
    app = build_graph()

    questions = [
        "What is RAG and why is it useful?",
        "How does Ollama help developers?",
        "What is the capital of France?",
    ]
    for q in questions:
        ask(app, q)

    print(f"\n{'='*60}")
    print("Interactive mode (type 'exit' to quit)\n")
    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input:
            ask(app, user_input)


if __name__ == "__main__":
    main()
