import chromadb

# First, loading a document:
def load_document(filepath):
    with open(filepath, "r") as f:
        return f.read()


# Second, splitting the data into chunks:
def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks
    

# Next, storing these chunks as "vectors" in ChromaDB:
def store_chunks(chunks):
    client = chromadb.Client()
    collection = client.create_collection("mini_rag")
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[str(i)]
        )
    return collection


# Lastly, searching by meaning and NOT keyword (semantic search):
def search(collection, query):
    results = collection.query(
        query_texts = [query],
        n_results = 2
    )
    return results["documents"][0]



text = load_document("sample.txt")
chunks = chunk_text(text)
collection = store_chunks(chunks)
results = search(collection, "Who is Batman?")
for r in results:
    print(r)
    print("---")