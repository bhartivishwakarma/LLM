from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


llm=OllamaLLM(model="gemma:2b",base_url="http://localhost:11434",temperature=0,streaming=False)

#langchain Runnables
template=PromptTemplate.from_template("Answer the following Question:{question}")
prompt=template.invoke({"question":"In which country is Nvidia's world headquater?"})
print(llm.invoke(prompt))
questions=[{"question":"In which country is Nvidia's world headquater?"},
           {"question":"When was Nvidia founded?"},
           {"question":"Who is the CEO of Nvidia?"}]
prompts=template.batch(questions)
print((prompts))
