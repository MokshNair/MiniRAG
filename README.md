# MiniRAG — Minimal RAG Pipeline from Scratch

A lightweight Retrieval Augmented Generation (RAG) pipeline built from scratch in Python. No LangChain, no abstractions — just the raw mechanics of how RAG works under the hood.

---

## What It Does

1. **Loads** a text document from disk
2. **Chunks** it into overlapping segments to preserve context
3. **Stores** chunks as vectors in ChromaDB using semantic embeddings
4. **Searches** by meaning — not keywords — to retrieve relevant chunks for any query

---

## Why I Built This

To understand exactly what happens inside a RAG system at the code level, without relying on framework abstractions. Every line is written and understood from first principles.

---

## Tech Stack

- Python — core logic
- ChromaDB — local vector database
- all-MiniLM-L6-v2 — local embedding model (no API cost)

---

## How To Run

**1. Install dependencies**
```bash
pip install chromadb
```

**2. Add your document**

Create a `sample.txt` file in the project folder with any text content.

**3. Run the pipeline**
```bash
python mini_rag.py
```

---

## Core Functions

| Function | What It Does |
|---|---|
| `load_document(filepath)` | Reads a text file and returns its content |
| `chunk_text(text, chunk_size, overlap)` | Splits text into overlapping chunks |
| `store_chunks(chunks)` | Embeds and stores chunks in ChromaDB |
| `search(collection, query)` | Semantic search — returns most relevant chunks |
