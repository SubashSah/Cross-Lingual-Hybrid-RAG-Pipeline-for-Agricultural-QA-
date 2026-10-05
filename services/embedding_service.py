from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "BAAI/bge-m3"


def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME
    )

    return embeddings