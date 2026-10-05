import pickle


with open(
    "storage/bm25/bm25_chunks.pkl",
    "rb"
) as f:
    bm25_chunks = pickle.load(f)

# Load original chunks

with open(
    "storage/chunks/original_chunks.pkl",
    "rb"
) as f:
    original_chunks = pickle.load(f)


chunk_index = 6



print("Original chunk:")
print(original_chunks[chunk_index].page_content)
print("BM25 chunk:")
print(bm25_chunks[chunk_index]["page_content"])
