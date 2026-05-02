from langchain_core.messages import SystemMessage, HumanMessage
from csv_visualizer.llm.cerebras_client import get_llm


def run_llm(system_prompt: str, user_prompt: str, temperature: float = 0):
    """
    Sends prompts to Cerebras and returns the model response.
    """

    llm = get_llm(temperature=temperature)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)

    return response.content