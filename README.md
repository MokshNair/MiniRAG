# MiniRAG — Minimal RAG Pipeline from Scratch

A lightweight Retrieval Augmented Generation (RAG) pipeline built from scratch in Python. No LangChain, no vector database, no abstractions — the actual math behind semantic search, written and understood line by line.

---

## What It Does

`MiniRag` is a single class that:

1. **Loads** a text document from disk
2. **Chunks** it into overlapping segments to preserve context across boundaries
3. **Embeds** each chunk into a real numeric vector using a pretrained sentence-transformer model
4. **Searches** two ways:
   - `search_chunk(keyword)` — naive substring/keyword matching
   - `semantic_search(query, top_n)` — real semantic search, ranking every chunk by cosine similarity to the query

---

## Why I Built This

To understand exactly what happens inside a RAG system at the code level — not just call `collection.query(...)` and trust it, but actually implement the comparison math (dot product, magnitude, cosine similarity) by hand and verify it against known values before trusting it on real text.

The one part **not** built from scratch is the embedding model itself — converting raw text into a meaningful vector requires a model trained on large amounts of text, which is a separate, much harder problem than the retrieval math. `sentence-transformers` handles that step; everything downstream of it (the actual comparison and ranking logic) is hand-built.

---

## Tech Stack

- Python — core logic, no LangChain
- `sentence-transformers` (`all-MiniLM-L6-v2`) — local embedding model, no API cost
- Hand-implemented dot product, magnitude, and cosine similarity — no numpy/scipy shortcuts

---

## How To Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add your document**

Create a `sample.txt` file in the project folder with any text content.

**3. Run the pipeline**
```bash
python mini_rag.py
```

---

## Core Methods

| Method | What It Does |
|---|---|
| `load_doc(filepath)` | Reads a text file, stores it on `self.text` |
| `chunk_doc(chunk_size, overlap)` | Splits text into overlapping chunks, embeds each one |
| `search_chunk(keyword)` | Naive keyword substring match across chunks |
| `semantic_search(query, top_n)` | Embeds the query, ranks all chunks by cosine similarity, returns top matches |
| `dot_product(a, b)` | Hand-built dot product between two equal-length vectors |
| `magnitude(a)` | Hand-built vector magnitude (Pythagorean theorem, N dimensions) |
| `cosine_similarity(a, b)` | Combines dot product and magnitude into the full similarity score |