from langchain_community.llms import OpenAI

def query_llm(prompt: str) -> str:
    llm = OpenAI(temperature=0.2)
    response = llm(prompt)
    return response
