import pickle

from services.embedding_service import get_embedding_model
from services.vector_store_service import load_vector_store


# Load FAISS
embedding_model = get_embedding_model()

vector_store = load_vector_store(
    "storage/faiss",
    embedding_model
)

faiss_chunks = list(
    vector_store.docstore._dict.values()
)


# Load original chunks

with open(
    "storage/chunks/original_chunks.pkl",
    "rb"
) as f:
    original_chunks = pickle.load(f)



print("FAISS chunks:", len(faiss_chunks))
print("Original chunks:", len(original_chunks))


for i in range(len(original_chunks)):

    if faiss_chunks[i].page_content != original_chunks[i].page_content:

        print("Mismatch found at:", i)

        print("FAISS:")
        print(faiss_chunks[i].page_content[:200])

        print("Original:")
        print(original_chunks[i].page_content[:200])

        break

else:

    print("All chunks are identical")