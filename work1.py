from langchain_ollama import OllamaLLM
llm=OllamaLLM(model="gemma:2b")
print(llm.invoke("who are you"))