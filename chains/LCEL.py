from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm=OllamaLLM(model="gemma:2b",base_url="http://localhost:11434",temperature=0,streaming=False)
template=PromptTemplate.from_template("Answer the following Question:{question}")
chain=template | llm 
print(chain.get_graph().draw_ascii())
answer=chain.invoke({"question":"Who founded Nvidia?"})
print(answer)