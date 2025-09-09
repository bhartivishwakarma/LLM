from langchain_ollama import OllamaLLM
llm=OllamaLLM(base_url="http://localhost:11434", model="gemma:2b",temperature=1)
prompt="Tell me a short fun fact about space"
result=llm.invoke(prompt)
print(result)

