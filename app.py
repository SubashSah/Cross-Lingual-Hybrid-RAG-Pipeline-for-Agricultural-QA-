from fastapi import FastAPI
from pydantic import BaseModel

from services.embedding_service import get_embedding_model
from services.loader_service import load_markdown
from services.vector_store_service import load_vector_store
from services.retriever_service import (
    create_semantic_retriever,
    create_keyword_retriever,
    hybrid_search,
)
from services.prompt_service import create_rag_prompt
from services.llm_service import get_llm
from services.retriever_service import load_bm25

VECTOR_STORE_PATH = "storage/faiss"


app = FastAPI(
    title="RAG Pipeline API",
    description="RAG pipeline using LangChain, FAISS and WiseAI vLLM",
    version="1.0.0",
)

embedding_model = get_embedding_model()

vector_store = load_vector_store(VECTOR_STORE_PATH, embedding_model)

semantic_retriever = create_semantic_retriever(vector_store, k=8)

bm25, bm25_chunks, original_chunks = load_bm25()


keyword_retriever = create_keyword_retriever(
    bm25,
    bm25_chunks,
    original_chunks
)


prompt = create_rag_prompt()


llm = get_llm()

# -----------------------------
# Load complete document globally
# -----------------------------
documents = load_markdown("data/tomato.md")
full_context = "\n\n".join(doc.page_content for doc in documents)


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.post("/chat")
def chat(request: ChatRequest):

    question = request.question

    # -----------------------------
    # 1. Semantic Search RAG
    # -----------------------------

    semantic_docs = semantic_retriever.invoke(question)

    # semantic_docs = rerank_documents(
    #     question,
    #     semantic_candidates,
    #     top_k=4
    # )

    semantic_context = "\n\n".join(
        doc.page_content
        for doc in semantic_docs
    )


    semantic_messages = prompt.invoke(
        {
            "context": semantic_context,
            "question": question
        }
    )

    semantic_response = llm.invoke(
        semantic_messages
    )


    # -----------------------------
    # 2. Hybrid Search RAG
    # -----------------------------

    hybrid_docs = hybrid_search(
        question,
        semantic_retriever,
        keyword_retriever,
        k=8,
        candidate_k=30,
    )

    # hybrid_docs = rerank_documents(
    #     question,
    #     hybrid_candidates,
    #     top_k=4
    # )

    hybrid_context = "\n\n".join(
        doc.page_content
        for doc in hybrid_docs
    )


    hybrid_messages = prompt.invoke(
        {
            "context": hybrid_context,
            "question": question
        }
    )


    hybrid_response = llm.invoke(
        hybrid_messages
    )


    # -----------------------------
    # 3. Full Document
    # -----------------------------

    full_messages = prompt.invoke(
        {
            "context": full_context,
            "question": question
        }
    )


    full_response = llm.invoke(
        full_messages
    )


    return {

        "question": question,

        "semantic_answer":
            semantic_response.content,

        "hybrid_answer":
            hybrid_response.content,

        "full_document_answer":
            full_response.content,

        "retrieved_documents": {

            "semantic": [
                doc.page_content
                for doc in semantic_docs
            ],

            "hybrid": [
                doc.page_content
                for doc in hybrid_docs
            ]

        }

    }
