from langchain_community.document_loaders import TextLoader


def load_markdown(file_path: str):
    """
    Load a Markdown document and return LangChain Document objects.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    return documents