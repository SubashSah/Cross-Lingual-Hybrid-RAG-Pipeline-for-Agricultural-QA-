import pickle

from services.normalization_service import normalize_scores



def create_semantic_retriever(vector_store, k):

    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    return retriever


def load_bm25():

    with open("storage/bm25/bm25.pkl", "rb") as f:

        bm25 = pickle.load(f)

    with open("storage/bm25/bm25_chunks.pkl", "rb") as f:

        bm25_chunks = pickle.load(f)

    with open("storage/chunks/original_chunks.pkl", "rb") as f:

        original_chunks = pickle.load(f)

    return bm25, bm25_chunks, original_chunks


def create_keyword_retriever(bm25, bm25_chunks, original_chunks, k=None):

    def retrieve(question, k=k, return_scores=False):

        target_k = k

        # tokenize question
        tokenized_query = question.split()

        # BM25 ranking
        scores = bm25.get_scores(tokenized_query)

        # top k indexes
        ranked_indexes = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:target_k]

        results = []

        for idx in ranked_indexes:

            chunk_id = bm25_chunks[idx]["metadata"]["chunk_id"]
            doc = original_chunks[chunk_id]

            if return_scores:
                results.append((doc, float(scores[idx])))
            else:
                results.append(doc)

        return results

    return retrieve



def hybrid_search(
    question,
    semantic_retriever,
    keyword_retriever,
    k,
    candidate_k,
    alpha=0.7,
    normalization_method="min_max",
    normalize_semantic=True,
    **kwargs,
):
    """
    Score-Based Fusion (SBF) of semantic and keyword retrievers.

    Args:
        question: The search query string.
        semantic_retriever: Semantic retriever or FAISS vector store.
        keyword_retriever: Keyword retriever callable.
        k: Number of final fused documents to return.
        candidate_k: Number of semantic and keyword candidates to retrieve
                     and normalize before fusion.
        alpha: Weight for semantic score (1 - alpha for keyword score). Default 0.7.
        normalization_method: 'min_max' or 'max' normalization for scores.
        normalize_semantic: Whether to also normalize semantic relevance scores
                            to [0, 1] for balanced weighting. Default True.
    """

    # --------------------------
    # 1. Dense (Semantic) Retrieval
    # --------------------------
    if hasattr(semantic_retriever, "vectorstore"):
        raw_candidates = semantic_retriever.vectorstore.similarity_search_with_score(
            question, k=candidate_k
        )
        # Convert Euclidean distance to relevance score in [0, 1]
        semantic_candidates = [
            (doc, max(0.0, min(1.0, 1.0 - float(dist) / 1.41421356)))
            for doc, dist in raw_candidates
        ]
    elif hasattr(semantic_retriever, "similarity_search_with_score"):
        raw_candidates = semantic_retriever.similarity_search_with_score(
            question, k=candidate_k
        )
        semantic_candidates = [
            (doc, max(0.0, min(1.0, 1.0 - float(dist) / 1.41421356)))
            for doc, dist in raw_candidates
        ]
    else:
        docs = semantic_retriever.invoke(question)
        semantic_candidates = [(doc, 1.0) for doc in docs[:candidate_k]]

    # Normalize semantic scores across top candidates if enabled
    if normalize_semantic and len(semantic_candidates) > 0:
        semantic_candidates = normalize_scores(
            semantic_candidates, method=normalization_method
        )

    # --------------------------
    # 2. Keyword (BM25) Retrieval
    # --------------------------
    keyword_candidates = keyword_retriever(
        question, k=candidate_k, return_scores=True
    )

    # Normalize keyword scores to [0, 1]
    normalized_keyword_candidates = normalize_scores(
        keyword_candidates, method=normalization_method
    )

    # --------------------------
    # 3. Score-Based Fusion (SBF)
    # --------------------------
    documents = {}
    fused_scores = {}

    for doc, score in semantic_candidates:
        doc_id = doc.metadata["chunk_id"]
        documents[doc_id] = doc
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (alpha * score)

    for doc, score in normalized_keyword_candidates:
        doc_id = doc.metadata["chunk_id"]
        documents[doc_id] = doc
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + ((1.0 - alpha) * score)

    # Rank documents by fused score in descending order
    ranked_documents = sorted(
        documents.values(),
        key=lambda doc: fused_scores[doc.metadata["chunk_id"]],
        reverse=True,
    )

    return ranked_documents[:k]
