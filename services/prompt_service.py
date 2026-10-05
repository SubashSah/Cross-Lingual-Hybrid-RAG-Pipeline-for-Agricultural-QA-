from langchain_core.prompts import ChatPromptTemplate


def create_rag_prompt():
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer the question using only the provided context.
If the answer cannot be found in the context, say that you
do not have enough information from the provided document.
Also, if the question is in a language other than English, answer in that language.

Context:
{context}

Question:
{question}

Answer:
""")

    return prompt
