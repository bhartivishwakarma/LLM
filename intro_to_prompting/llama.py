from langchain_ollama import OllamaLLM
llm= OllamaLLM(model="gemma:2b", temperature=0.1, max_tokens=20)
print(llm.invoke("Tell me a short fun fact about space"))