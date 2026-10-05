from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-v2-m3"


reranker_model = CrossEncoder(
    MODEL_NAME,
    max_length=512
)


def rerank_documents(
    question,
    documents,
    top_k=4
):

    if not documents:
        return []


    pairs = []

    for doc in documents:

        pairs.append(
            [
                question,
                doc.page_content
            ]
        )


    scores = reranker_model.predict(
        pairs
    )


    scored_documents = list(
        zip(
            documents,
            scores
        )
    )


    ranked_documents = sorted(
        scored_documents,
        key=lambda x:x[1],
        reverse=True
    )


    reranked_documents = [
        doc
        for doc, score in ranked_documents[:top_k]
    ]


    return reranked_documents