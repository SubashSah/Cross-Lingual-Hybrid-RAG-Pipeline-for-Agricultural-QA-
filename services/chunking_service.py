from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=300,
    )

    chunks = recursive_splitter.split_documents(documents)

    # assign stable chunk ids after final splitting
    for idx, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = idx

    return chunks