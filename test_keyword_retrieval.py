from services.retriever_service import load_bm25
from services.retriever_service import create_keyword_retriever

bm25, bm25_chunks, original_chunks = load_bm25()

keyword_retriever = create_keyword_retriever(bm25, bm25_chunks, original_chunks, k=4)

keyword_documents = keyword_retriever("टमाटर grafting को मुख्य उद्देश्य के हो?")

print("Retrieved documents:")
for doc in keyword_documents:
    print(doc.page_content)
    print("----"*10)