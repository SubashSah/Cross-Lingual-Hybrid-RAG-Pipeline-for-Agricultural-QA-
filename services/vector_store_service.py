from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):
    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    return vector_store


def save_vector_store(vector_store, path):
    vector_store.save_local(path)
    

def load_vector_store(path, embedding_model):
    vector_store = FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_store