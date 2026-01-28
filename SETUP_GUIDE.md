# LangChain RAG Project Setup Guide

> Build a Retrieval Augmented Generation (RAG) system using LangChain, ChromaDB, and Groq.

**Prepared by: Salman Zaman** | Certified Google Cloud Generative AI Leader | Certified Chief AI Officer (CAIO) | Certified Google Cloud Digital Leader | Certified Mastery in Data Science | Certified Information Systems Auditor (CISA) | Certified IBM OnDemand Business Solution Designer | Certified Java Programmer | Certified IBM DB2 Enterprise Content Manager | Certified IBM Lotus Notes/Domino Application Developer

---

## Prerequisites

Before starting, ensure you have:

- **Operating System:** Windows 10/11, macOS, or Linux
- **VS Code** installed ([download here](https://code.visualstudio.com/))
- **uv** installed ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Groq API Key** (free at https://console.groq.com)

---

# PART 1: Project Initialization

---

## Step 1: Create Project Folder

Create a folder for your project and open it in VS Code.

**Windows (CMD / PowerShell):**
```cmd
mkdir langchain_rag
cd langchain_rag
code .
```

**Linux / Bash / Git Bash:**
```bash
mkdir langchain_rag
cd langchain_rag
code .
```

**macOS:**
```bash
mkdir langchain_rag
cd langchain_rag
code .
```

**Alternative:** Open VS Code → **File** → **Open Folder** → select/create your folder.

---

## Step 2: Initialize Project with uv

Open terminal in VS Code (`Ctrl + `` ` on Windows/Linux, `Cmd + `` ` on macOS) and run:

**All Platforms:**
```bash
uv init
```

This creates the following files:
- `pyproject.toml` - Project configuration
- `main.py` - Main Python file
- `.python-version` - Python version specification
- `README.md` - Project readme

---

## Step 3: Get Groq API Key

1. Go to https://console.groq.com
2. Sign up / Log in
3. Click **API Keys** → **Create API Key**
4. Copy and save the key immediately

> **Important:** Copy your API key right away. You won't be able to see it again after closing the dialog!

---

# PART 2: Configuration

---

## Step 4: Create .env File

1. In VS Code, create a new file named `.env` in your project root
2. Add your API key:

```env
GROQ_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual Groq API key.

> **Note:** The `.env` file stores sensitive information. Never share this file or commit it to Git!

---

## Step 5: Create .gitignore

Create a file named `.gitignore` in your project root with the following content:

```gitignore
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info

# Virtual environments
.venv

# Environment variables (secrets)
.env
.env.local
.env.*.local

# Vector Stores
chroma_db/
```

> **Security Warning:** Never commit your `.env` file to Git. API keys in public repositories can be stolen and misused!

---

## Step 6: Install Dependencies

Run this command to install all required packages:

**All Platforms:**
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

Create a folder to store your documents:

**Windows (CMD / PowerShell):**
```cmd
mkdir data
```

**Linux / Bash / Git Bash / macOS:**
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

# PART 3: Running the Project

---

## Step 9: Run the Project

**All Platforms:**
```bash
uv run main.py
```

**What happens on first run:**
1. Downloads the embedding model (~90MB) - this only happens once
2. Loads your documents (text, PDF, web)
3. Creates embeddings and stores them in ChromaDB
4. Starts the interactive Q&A session

Subsequent runs are faster since the model is cached.

---

## Step 10: Interact with Your RAG System

Once running, you can ask questions about your documents:

```
You: What is attention in transformers?
AI: [Answer based on your documents]

You: What are autonomous agents?
AI: [Answer based on the web document]
```

---

## Step 11: Exit the Application

Type `quit`, `exit`, or `q` to stop the program.

Or press `Ctrl + C` to force quit.

---

## Project Structure

After completing all steps, your project should look like this:

```
langchain_rag/
├── .env                 # API key (DO NOT COMMIT)
├── .gitignore           # Git ignore rules
├── .python-version      # Python version
├── .venv/               # Virtual environment (created by uv)
├── chroma_db/           # Vector database (created on first run)
├── data/
│   ├── text_file.txt    # Your text content
│   └── attention.pdf    # Transformer paper (or any PDF)
├── main.py              # Main application
├── pyproject.toml       # Project config
├── README.md            # Project readme
└── uv.lock              # Dependency lock file
```

**Data Sources Summary:**
| Source | Location/URL |
|--------|--------------|
| Text | `./data/text_file.txt` |
| PDF | `./data/attention.pdf` |
| Web | `https://lilianweng.github.io/posts/2023-06-23-agent/` |

---

# PART 4: GitHub Lab

Push your project to GitHub securely and learn essential Git commands.

---

## Step 12: Create a GitHub Account

If you don't have a GitHub account, follow these steps:

1. Go to https://github.com
2. Click **Sign up**
3. Enter your email address
4. Create a password (min 8 characters, include a number and lowercase letter)
5. Choose a username (this will be your public identity)
6. Complete the verification puzzle
7. Verify your email by clicking the link sent to your inbox

> **Tip:** Choose a professional username - it will appear in all your repository URLs.

---

## Step 13: Install Git

### Windows

**Option 1: Download Installer (Recommended for beginners)**
1. Go to: https://git-scm.com/download/win
2. Download will start automatically
3. Run the installer
4. Use default options - click Next through all steps
5. Restart VS Code after installation

**Option 2: Using winget (PowerShell)**
```powershell
winget install Git.Git
```

### Linux

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install git
```

**Fedora:**
```bash
sudo dnf install git
```

**Arch Linux:**
```bash
sudo pacman -S git
```

### macOS

**Option 1: Using Xcode Command Line Tools (Recommended)**
```bash
xcode-select --install
```

**Option 2: Using Homebrew**
```bash
brew install git
```

---

## Step 14: Verify Git Installation

After installation, verify Git is working:

**All Platforms:**
```bash
git --version
```

You should see something like: `git version 2.43.0`

If you get "command not found", restart your terminal or VS Code.

---

## Step 15: Configure Git Identity

Set your name and email (use the same email as your GitHub account):

**All Platforms:**
```bash
git config --global user.name "Your Full Name"
git config --global user.email "your.email@example.com"
```

Verify your configuration:

```bash
git config --list
```

You should see your name and email in the output.

---

## Step 16: Create a New Repository on GitHub

1. Go to https://github.com/new
2. **Repository name:** `langchain-rag`
3. **Description:** "RAG pipeline with LangChain, ChromaDB, and Groq"
4. Select **Public** (or Private if you prefer)
5. **DO NOT** check "Add a README" (we already have files)
6. Click **Create repository**

---

## Step 17: Initialize and Push to GitHub

> **Security Check:** Before pushing, ensure your `.gitignore` file includes `.env` to prevent exposing your API key!

In VS Code terminal, run these commands (in your project folder):

**All Platforms:**
```bash
# Step 1: Initialize git repository
git init

# Step 2: Add all files (respects .gitignore)
git add .

# Step 3: Create first commit
git commit -m "Initial commit: LangChain RAG pipeline"

# Step 4: Rename branch to main
git branch -M main

# Step 5: Connect to your GitHub repository (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/langchain-rag.git

# Step 6: Push to GitHub
git push -u origin main
```

> **First Push:** GitHub will prompt you to authenticate. Sign in with your browser when prompted, or use a Personal Access Token.

---

## Step 18: Verify Your Push

1. Go to `https://github.com/YOUR_USERNAME/langchain-rag`
2. You should see all your project files

> **Security Confirmation:** Check that `.env` is NOT visible in your GitHub repository. If it is, remove it immediately and regenerate your API key!

---

## Essential Git Commands Reference

### Daily Workflow Commands

| Command | Purpose |
|---------|---------|
| `git status` | See changed/staged files |
| `git add .` | Stage all changes |
| `git add filename` | Stage specific file |
| `git commit -m "message"` | Commit staged changes |
| `git push` | Upload commits to GitHub |
| `git pull` | Download latest from GitHub |

### Viewing History & Changes

| Command | Purpose |
|---------|---------|
| `git log --oneline` | View commit history (compact) |
| `git log` | View detailed commit history |
| `git diff` | See unstaged changes |
| `git diff --staged` | See staged changes |

### Branching (For Advanced Workflows)

| Command | Purpose |
|---------|---------|
| `git branch` | List all branches |
| `git branch feature-name` | Create new branch |
| `git checkout feature-name` | Switch to branch |
| `git checkout -b feature-name` | Create and switch in one step |
| `git merge feature-name` | Merge branch into current |

### Undoing Changes

| Command | Purpose |
|---------|---------|
| `git checkout -- filename` | Discard changes in file |
| `git reset HEAD filename` | Unstage a file |
| `git reset --soft HEAD~1` | Undo last commit (keep changes) |

---

## Typical Update Workflow

When you make changes to your project:

**All Platforms:**
```bash
# 1. Check what changed
git status

# 2. Stage your changes
git add .

# 3. Commit with a descriptive message
git commit -m "Add feature: support for CSV file loading"

# 4. Push to GitHub
git push
```

---

## Cloning Your Repo on Another Machine

If you need to work on your project from a different computer:

**All Platforms:**
```bash
git clone https://github.com/YOUR_USERNAME/langchain-rag.git
cd langchain-rag
```

> **Remember:** After cloning, create a new `.env` file with your API key - it won't be in the repo!

---

## Git Security Best Practices

- **Never commit secrets:** Always use `.gitignore` for `.env`, API keys, passwords
- **Review before pushing:** Run `git status` and `git diff` before committing
- **Use meaningful commits:** Write clear messages describing what changed
- **If you accidentally push secrets:** Immediately revoke/regenerate the exposed key

---

# Troubleshooting

| Problem | Solution |
|---------|----------|
| `uv is not recognized` | Restart terminal or reinstall uv |
| `No module named X` | Run `uv add <package-name>` |
| `GROQ_API_KEY not found` | Check `.env` file exists with correct format |
| `API key invalid` | Get new key from https://console.groq.com/keys |
| `Connection error` | Check internet connection |
| `git is not recognized` | Restart terminal/VS Code after Git installation |
| `Permission denied (publickey)` | Use HTTPS URL instead of SSH, or set up SSH keys |
| Slow first run | Normal - downloading embedding model (~90MB) |

---

# Quick Reference

## uv Commands

| Action | Command |
|--------|---------|
| Initialize project | `uv init` |
| Add package | `uv add package-name` |
| Run program | `uv run main.py` |
| List packages | `uv pip list` |
| Remove package | `uv remove package-name` |

## Git Commands

| Action | Command |
|--------|---------|
| Initialize repo | `git init` |
| Stage all files | `git add .` |
| Commit changes | `git commit -m "message"` |
| Push to remote | `git push` |
| Pull from remote | `git pull` |
| Check status | `git status` |

---

# Customization Options

- **chunk_size**: Smaller = more precise, larger = more context (default: 1000)
- **chunk_overlap**: Overlap between chunks to preserve context (default: 200)
- **temperature**: 0.0 = factual/deterministic, 1.0 = creative (default: 0.0)
- **k**: Number of chunks retrieved per query (default: 3)

---

*Built with LangChain, ChromaDB, HuggingFace, and Groq*

*Prepared by: Salman Zaman for the ambitious Software Orchestrators / Architects!*
