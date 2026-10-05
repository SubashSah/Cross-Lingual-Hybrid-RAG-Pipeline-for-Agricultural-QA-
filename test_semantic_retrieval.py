import sys
from services.embedding_service import get_embedding_model
from services.vector_store_service import load_vector_store
from services.retriever_service import create_semantic_retriever

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

VECTOR_STORE_PATH = "storage/faiss"

embedding_model = get_embedding_model()
vector_store = load_vector_store(VECTOR_STORE_PATH, embedding_model)

semantic_retriever = create_semantic_retriever(vector_store, k=4)

semantic_documents = semantic_retriever.invoke("टमाटर grafting को मुख्य उद्देश्य के हो?")

print("Retrieved documents:")
for doc in semantic_documents:
    print(doc.page_content)
    print("----" * 10)
