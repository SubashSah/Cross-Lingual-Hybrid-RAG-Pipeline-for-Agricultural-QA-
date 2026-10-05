import pickle
import os

from rank_bm25 import BM25Okapi


from services.loader_service import load_markdown
from services.chunking_service import chunk_documents
from services.vector_store_service import (
    create_vector_store,
    save_vector_store
)
from services.embedding_service import get_embedding_model
from services.translation_service import translate_chunks



FILE_PATH = "data/tomato.md"



def create_bm25_index(translated_chunks):

    documents = [
        chunk["page_content"]
        for chunk in translated_chunks
    ]

    tokenized_documents = [
        doc.split()
        for doc in documents
    ]


    bm25 = BM25Okapi(
        tokenized_documents
    )


    return bm25



def main():

    # ==========================
    # Load document
    # ==========================

    documents = load_markdown(
        FILE_PATH
    )


    # ==========================
    # Create English chunks
    # ==========================

    chunks = chunk_documents(
        documents
    )


    print(
        "Total chunks:",
        len(chunks)
    )


    # ==========================
    # Save original chunks
    # ==========================

    os.makedirs(
        "storage/chunks",
        exist_ok=True
    )


    with open(
        "storage/chunks/original_chunks.pkl",
        "wb"
    ) as f:

        pickle.dump(
            chunks,
            f
        )


    print(
        "Original English chunks saved"
    )



    # ==========================
    # Create FAISS
    # ==========================

    embedding_model = get_embedding_model()


    vector_store = create_vector_store(
        chunks,
        embedding_model
    )


    save_vector_store(
        vector_store,
        "storage/faiss"
    )


    print(
        "FAISS saved"
    )



    # ==========================
    # Translate chunks
    # ==========================

    translated_chunks = translate_chunks(
        chunks
    )


    print(
        "Translation completed"
    )



    # ==========================
    # Create BM25
    # ==========================

    bm25 = create_bm25_index(
        translated_chunks
    )



    os.makedirs(
        "storage/bm25",
        exist_ok=True
    )



    # Save BM25 model

    with open(
        "storage/bm25/bm25.pkl",
        "wb"
    ) as f:

        pickle.dump(
            bm25,
            f
        )



    # Save Nepali chunks with ids

    with open(
        "storage/bm25/bm25_chunks.pkl",
        "wb"
    ) as f:

        pickle.dump(
            translated_chunks,
            f
        )



    print(
        "BM25 saved"
    )



if __name__ == "__main__":
    main()

    