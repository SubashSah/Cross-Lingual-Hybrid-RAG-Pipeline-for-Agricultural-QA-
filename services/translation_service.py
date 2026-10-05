from langchain_core.prompts import ChatPromptTemplate
from services.llm_service import get_llm

llm = get_llm()

translation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional translator. Translate the following English text into Nepali (Devanagari script).\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Output ONLY the direct translated Nepali text.\n"
            "- Do NOT include any greetings, notes, explanations, phonetic transliterations, or conversational commentary.\n"
            "- Maintain all Markdown formatting, headings, bullet points, numbers, and lists exactly as in the original text.",
        ),
        (
            "user",
            "{text}",
        ),
    ]
)


def translate_text(text: str) -> str:
    """
    Translate an entire text chunk from English to Nepali using the configured LLM.
    """
    if not text or not text.strip():
        return ""

    # Normalize dash characters
    cleaned_text = text.replace("–", "-")

    messages = translation_prompt.invoke({"text": cleaned_text})
    response = llm.invoke(messages)
    return response.content.strip()


def translate_chunks(chunks):
    """
    Translate LangChain chunks into Nepali using the configured LLM,
    preserving chunk_id and metadata mapping.
    """
    translated_chunks = []

    for idx, chunk in enumerate(chunks):
        print(f"Translating chunk {idx + 1}/{len(chunks)} with LLM...")

        translated_text = translate_text(chunk.page_content)

        translated_chunks.append(
            {
                "page_content": translated_text,
                "metadata": {
                    **chunk.metadata,
                },
            }
        )

    return translated_chunks