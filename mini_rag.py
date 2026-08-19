import math
from sentence_transformers import SentenceTransformer

class MiniRag:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_doc(self, filepath):
        # store filepath and text on self so other methods can reuse self.text
        self.filepath = filepath
        with open(filepath, "r") as f:
            self.text = f.read()
            return self.text


    def chunk_doc(self, chunk_size=150, overlap=10):
        # step forward by (chunk_size - overlap) instead of chunk_size, so consecutive chunks share some text
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []
        self.chunk_embeddings = []  # parallel list - same index as self.chunks, holds each chunk's vector
        i = 0
        while i < len(self.text):
            chunk = self.text[i:i+self.chunk_size]
            self.chunks.append(chunk)
            self.chunk_embeddings.append(self.model.encode(chunk))  # real embedding, not a toy vector
            i += (self.chunk_size - self.overlap)
        return self.chunks


    def search_chunk(self, keyword):
        # dumb keyword match (substring check), not semantic - cosine_similarity below is the smart version
        self.keyword = keyword
        self.results = []
        for chunk in self.chunks:
            if self.keyword in chunk:
                self.results.append(chunk)
        return self.results

    def semantic_search(self, query, top_n=2):
        # embed the query the same way chunks were embedded, so they're comparable
        query_embedding = self.model.encode(query)
        chunk_score = []
        for chunk, embedding in zip(self.chunks, self.chunk_embeddings):
            # pair each chunk with its similarity score to the query
            chunk_score.append((chunk, self.cosine_similarity(query_embedding, embedding)))
        # sort() returns a new list, doesn't sort in place - must capture the result
        sorted_list = sorted(chunk_score, key=lambda x:x[1], reverse=True)
        return sorted_list[:top_n]


    def dot_product(self, a, b):
        # multiply matching positions, sum the products - only valid if a and b are the same length
        products = []
        if len(a) == len(b):
            i = 0
            while i < len(a):
                product = a[i] * b[i]
                products.append(product)
                i += 1
        dot_prod = sum(products)
        return dot_prod

    def magnitude(self, a):
        # vector length: sqrt of the sum of squares (Pythagorean theorem, N dimensions)
        running_sum = 0
        for i in a:
            running_sum += (i*i)
        sqrt = math.sqrt(running_sum)
        return sqrt

    def cosine_similarity(self, a, b):
        # dot_product / (magnitude * magnitude) - how similar two vectors' directions are, close to 1 = similar
        dot_pro = self.dot_product(a, b)
        mag_a = self.magnitude(a)
        mag_b = self.magnitude(b)
        cos_sim = dot_pro/(mag_a * mag_b)
        return cos_sim


if __name__ == "__main__":
    rag = MiniRag()
    rag.load_doc("sample.txt")
    rag.chunk_doc()
    results = rag.semantic_search("Who is Batman?")
    for chunk, score in results:
        print(score, "-", chunk)
        print("---")
