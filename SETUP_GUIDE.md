# LangChain RAG Project Setup Guide

> Build a Retrieval Augmented Generation (RAG) system using LangChain, ChromaDB, and Groq.

**Prepared by: Salman Zaman** | Certified Google Cloud Generative AI Leader | Certified Chief AI Officer (CAIO) | Certified Google Cloud Digital Leader | Certified Data Scientist | ISACA Certified Information Systems Auditor (CISA) | Certified IBM OnDemand Business Solution Designer | Certified IBM Enterprise Content Manager

---

## Prerequisites

- Windows 10/11 (or macOS/Linux with minor adjustments)
- VS Code installed
- uv installed ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- Groq API Key (free at https://console.groq.com)

---

## Step 1: Create Project Folder

Create a folder for your project and open it in VS Code:

```bash
mkdir langchain_rag
cd langchain_rag
code .
```

Or open VS Code → **File** → **Open Folder** → select your folder.

---

## Step 2: Initialize Project with uv

Open terminal in VS Code (`Ctrl + `` `) and run:

```bash
uv init
```

This creates: `pyproject.toml`, `main.py`, `.python-version`, `README.md`

---

## Step 3: Get Groq API Key

1. Go to https://console.groq.com
2. Sign up / Log in
3. Click **API Keys** → **Create API Key**
4. Copy and save the key (you won't see it again)

---

## Step 4: Create .env File

1. Create a file named `.env` in project root
2. Add your API key:

```
GROQ_API_KEY=your_api_key_here
```

---

## Step 5: Create .gitignore

Create `.gitignore` file with:

```
# Python
__pycache__/
*.py[oc]
.venv/

# Secrets
.env
.env.local
```

---

## Step 6: Install Dependencies

```bash
uv add langchain langchain-groq langchain-community langchain-text-splitters langchain-huggingface langchain-chroma pypdf beautifulsoup4 lxml python-dotenv
```

| Package | Purpose |
|---------|---------|
| `langchain` | Core framework |
| `langchain-groq` | Groq LLM integration |
| `langchain-community` | Document loaders |
| `langchain-text-splitters` | Text chunking |
| `langchain-huggingface` | Embeddings |
| `langchain-chroma` | Vector store |
| `pypdf` | PDF reading |
| `beautifulsoup4`, `lxml` | Web scraping |
| `python-dotenv` | Load .env file |

---

## Step 7: Create Data Folder and Add Documents

```bash
mkdir data
```

Add these files to the `data/` folder:

### 7.1: text_file.txt
Create a text file with any content you want to query. Example: copy-paste an article about RAG systems or any topic of interest.

### 7.2: attention.pdf
Download the famous "Attention Is All You Need" paper (the Transformer paper):
- URL: https://arxiv.org/pdf/1706.03762
- Save as: `data/attention.pdf`

Or use any PDF document you want to query.

### 7.3: Web Document (already configured in code)
The code loads this blog post about LLM-powered autonomous agents:
- URL: https://lilianweng.github.io/posts/2023-06-23-agent/
- To change it, modify the `web_paths` in `main.py`

---

## Step 8: Replace main.py

Replace the contents of `main.py` with:

```python
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
```

---

## Step 9: Run the Project

```bash
uv run main.py
```

First run downloads the embedding model (~90MB). Subsequent runs are faster.

---

## Step 10: Exit

Type `quit`, `exit`, or press `Ctrl + C`.

---

## Project Structure

```
langchain_rag/
├── .env                 # API key (DO NOT COMMIT)
├── .gitignore           # Git ignore rules
├── .python-version      # Python version
├── .venv/               # Virtual environment
├── chroma_db/           # Vector database (created on run)
├── data/
│   ├── text_file.txt    # Your text content
│   └── attention.pdf    # Transformer paper (or any PDF)
├── main.py              # Main application
├── pyproject.toml       # Project config
└── uv.lock              # Dependency lock
```

**Data Sources:**
| Source | Location/URL |
|--------|--------------|
| Text | `./data/text_file.txt` |
| PDF | `./data/attention.pdf` |
| Web | `https://lilianweng.github.io/posts/2023-06-23-agent/` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `uv is not recognized` | Restart terminal or reinstall uv |
| `No module named X` | Run `uv add <package-name>` |
| `GROQ_API_KEY not found` | Check `.env` file exists with correct format |
| `API key invalid` | Get new key from https://console.groq.com/keys |
| `Connection error` | Check internet connection |
| Slow first run | Normal - downloading embedding model |

---

## Quick Reference

| Action | Command |
|--------|---------|
| Initialize project | `uv init` |
| Add package | `uv add package-name` |
| Run program | `uv run main.py` |
| List packages | `uv pip list` |
| Remove package | `uv remove package-name` |

---

## Customization

- **chunk_size**: Smaller = precise, larger = more context
- **temperature**: 0.0 = factual, 1.0 = creative
- **k**: Number of chunks retrieved (default: 3)

---

*Built with LangChain, ChromaDB, HuggingFace, and Groq*

*Prepared by: Salman Zaman for the ambitious Software Development Orchestrators/Architects!*
