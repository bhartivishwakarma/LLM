from langchain_ollama import OllamaLLM
llm=OllamaLLM(base_url="http://localhost:11434",model="gemma:2b",temperature=0.5)
prompt="when and where was NVIDIA founded in brief"
for chunk in llm.stream(prompt):
    print(chunk, end=" ",flush=True)