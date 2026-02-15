from langchain_ollama import OllamaLLM

llm=OllamaLLM(model="gemma:2b",temperature=0,max_tokens=1000)
one_off_prompt="Translate the following from English to Odia:Today is a good day"
print(llm.invoke(one_off_prompt))
