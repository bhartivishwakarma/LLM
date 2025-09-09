from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
llm=OllamaLLM(model="gemma:2b",temperature=0)
