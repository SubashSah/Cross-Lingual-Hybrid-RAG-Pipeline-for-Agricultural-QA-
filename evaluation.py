import json
import pandas as pd


from services.embedding_service import get_embedding_model
from services.vector_store_service import load_vector_store

from services.retriever_service import (
    create_semantic_retriever,
    create_keyword_retriever,
    hybrid_search,
    load_bm25,
)

from services.loader_service import load_markdown
from services.prompt_service import create_rag_prompt
from services.llm_service import get_llm

# -----------------------------
# Load RAG components
# -----------------------------

VECTOR_STORE_PATH = "storage/faiss"


embedding_model = get_embedding_model()


vector_store = load_vector_store(VECTOR_STORE_PATH, embedding_model)


# Semantic retriever (FAISS + BGE-M3)

semantic_retriever = create_semantic_retriever(vector_store, k=8)


# BM25 retriever (Nepali BM25 -> English chunks)

bm25, bm25_chunks, original_chunks = load_bm25()


keyword_retriever = create_keyword_retriever(bm25, bm25_chunks, original_chunks)


prompt = create_rag_prompt()


llm = get_llm()


# -----------------------------
# Load complete document
# -----------------------------

documents = load_markdown("data/tomato.md")


full_context = "\n\n".join(doc.page_content for doc in documents)


# -----------------------------
# Load questions
# -----------------------------

with open("questions.json", "r", encoding="utf-8") as f:

    questions = json.load(f)


results = []


# -----------------------------
# Run evaluation
# -----------------------------

for filename, question in questions.items():

    print("Processing:", filename)

    # =========================
    # Semantic Retrieval
    # =========================

    semantic_docs = semantic_retriever.invoke(question)

    # semantic_docs = rerank_documents(question, semantic_candidates, top_k=4)

    semantic_context = "\n\n".join(doc.page_content for doc in semantic_docs)

    semantic_prompt = prompt.invoke({"context": semantic_context, "question": question})

    semantic_answer = llm.invoke(semantic_prompt).content

    # =========================
    # Hybrid Retrieval
    # =========================

    hybrid_docs = hybrid_search(
        question,
        semantic_retriever,
        keyword_retriever,
        k=8,
        candidate_k=30,
    )

    # hybrid_docs = rerank_documents(question, hybrid_candidates, top_k=4)

    hybrid_context = "\n\n".join(doc.page_content for doc in hybrid_docs)

    hybrid_prompt = prompt.invoke({"context": hybrid_context, "question": question})

    hybrid_answer = llm.invoke(hybrid_prompt).content

    # =========================
    # Full Document
    # =========================

    full_prompt = prompt.invoke({"context": full_context, "question": question})

    full_answer = llm.invoke(full_prompt).content

    results.append(
        {
            "question": question,
            "semantic_retrieval": semantic_answer,
            "hybrid_retrieval": hybrid_answer,
            "whole_document": full_answer,
            # "semantic_docs": [doc.page_content for doc in semantic_docs],
            # "hybrid_docs": [doc.page_content for doc in hybrid_docs],
        }
    )


# -----------------------------
# Save CSV
# -----------------------------


df = pd.DataFrame(results)


df.to_csv("rag_evaluation_results.csv", index=False, encoding="utf-8-sig")


print("Evaluation completed!")
