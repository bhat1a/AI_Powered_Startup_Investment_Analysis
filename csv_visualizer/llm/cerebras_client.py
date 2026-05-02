import os
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras

load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
MODEL = os.getenv("MODEL", "llama3.1-70b")


def get_llm(temperature: float = 0):
    """
    Returns a Cerebras Chat model instance.
    """

    return ChatCerebras(
        api_key=CEREBRAS_API_KEY,
        model=MODEL,
        temperature=temperature
    )