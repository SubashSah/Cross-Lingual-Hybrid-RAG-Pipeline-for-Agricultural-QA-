import os 

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm():
    llm = ChatOpenAI(
        base_url=os.getenv("VLLM_BASE_URL"),
        api_key = "dummy",
        model=os.getenv("VLLM_MODEL"),
        temperature=0.0
    )
    return llm