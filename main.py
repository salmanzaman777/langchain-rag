"""
LangChain RAG Pipeline
======================
A complete Retrieval Augmented Generation system that:
1. Loads documents from multiple sources (Text, PDF, Web)
2. Cleans and chunks the text
3. Creates vector embeddings
4. Stores them in ChromaDB
5. Answers questions using Llama 3.1 via Groq
"""

import os
import re
from dotenv import load_dotenv

# Set user agent before importing web loaders (required by some websites)
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# LangChain imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def clean_text(text):
    """Remove extra whitespace and normalize text."""
    return re.sub(r'\s+', ' ', text).strip()


def get_clean_docs(docs):
    """Clean all documents by removing extra whitespace."""
    if not docs:
        print("[ERROR] No documents loaded!")
        return docs

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    return docs


def main():
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    print("=" * 60)
    print("LangChain RAG Pipeline")
    print("=" * 60)

    # Load API key from .env file
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("[ERROR] GROQ_API_KEY not found in .env file!")
        print("Please create a .env file with: GROQ_API_KEY=your_key_here")
        return

    print("[OK] API Key loaded from .env file")

    # =========================================================================
    # STEP 1: DATA INGESTION (Text, PDF, Web)
    # =========================================================================

    # 1a. Load Text Document
    print("\n" + "-" * 60)
    print("STEP 1(a): Loading Text documents...")
    print("-" * 60)
    text_docs = TextLoader("./data/text_file.txt").load()
    print(f"[OK] Loaded {len(text_docs)} TEXT document(s) | {len(text_docs[0].page_content)} characters")

    # 1b. Load PDF Document
    print("\n" + "-" * 60)
    print("STEP 1(b): Loading PDF documents...")
    print("-" * 60)
    pdf_docs = PyPDFLoader("./data/attention.pdf").load()
    print(f"[OK] Loaded {len(pdf_docs)} PDF page(s) | Page 1: {len(pdf_docs[0].page_content)} characters")

    # 1c. Load Web Document
    print("\n" + "-" * 60)
    print("STEP 1(c): Loading WEB documents...")
    print("-" * 60)
    web_docs = WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",)).load()
    print(f"[OK] Loaded {len(web_docs)} WEB document(s) | {len(web_docs[0].page_content)} characters")

    # =========================================================================
    # STEP 2: CONTENT CLEANING
    # =========================================================================
    print("\n" + "-" * 60)
    print("STEP 2: Cleaning documents...")
    print("-" * 60)

    text_docs = get_clean_docs(text_docs)
    pdf_docs = get_clean_docs(pdf_docs)
    web_docs = get_clean_docs(web_docs)

    print(f"[OK] Text: {len(text_docs[0].page_content)} | PDF Page 1: {len(pdf_docs[0].page_content)} | Web: {len(web_docs[0].page_content)} chars")

    # =========================================================================
    # STEP 3: CHUNKING
    # =========================================================================
    print("\n" + "-" * 60)
    print("STEP 3: Splitting into chunks...")
    print("-" * 60)

    # Split documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Max characters per chunk
        chunk_overlap=200,    # Overlap to preserve context between chunks
        separators=["\n\n", "\n", " ", ""]
    )

    text_chunks = text_splitter.split_documents(text_docs)
    pdf_chunks = text_splitter.split_documents(pdf_docs)
    web_chunks = text_splitter.split_documents(web_docs)

    print(f"[OK] Created {len(text_chunks)} TEXT | {len(pdf_chunks)} PDF | {len(web_chunks)} WEB chunks")

    # =========================================================================
    # STEP 4: CREATE EMBEDDINGS & VECTOR STORE
    # =========================================================================
    print("\n" + "-" * 60)
    print("STEP 4: Creating embeddings and vector store...")
    print("-" * 60)
    print("[INFO] This may take a minute on first run (downloading model)...")

    # Create embeddings using a local HuggingFace model (free, runs locally)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Combine all chunks and store in ChromaDB vector database
    all_chunks = text_chunks + pdf_chunks + web_chunks
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )
    print(f"[OK] Vector store created with {len(all_chunks)} chunks at ./chroma_db")

    # =========================================================================
    # STEP 5: SET UP THE RAG CHAIN
    # =========================================================================
    print("\n" + "-" * 60)
    print("STEP 5: Setting up RAG chain...")
    print("-" * 60)

    # Initialize LLM (Groq provides fast inference for Llama models)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
    print("[OK] Llama 3.1 via Groq LLM initialized.")

    # Retriever fetches top-k most relevant chunks for a query
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Helper to format retrieved documents into a single string
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # Prompt template: instructs the LLM how to use the retrieved context
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert technical assistant. "
         "Use the following pieces of retrieved context to answer the user's question. "
         "If the answer is not in the context, say you don't know.\n\n"
         "{context}"),
        ("human", "{question}"),
    ])

    # Build RAG chain using LCEL (LangChain Expression Language)
    # Flow: query -> retrieve docs -> format -> prompt -> LLM -> parse output
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("[OK] RAG chain ready!")

    # =========================================================================
    # STEP 6: INTERACTIVE QUERY LOOP
    # =========================================================================
    print("\n" + "-" * 60)
    print("STEP 6: Ready to answer questions!")
    print("-" * 60)
    print("Type your question and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print()

    while True:
        query = input("You: ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not query:
            continue

        print("\nThinking...\n")
        response = rag_chain.invoke(query)
        print(f"AI: {response}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
