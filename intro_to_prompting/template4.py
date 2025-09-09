from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
llm=OllamaLLM(model="gemma:2b", temperature=0.1, max_tokens=1000)
E_to_S_temp=ChatPromptTemplate.from_template("""Translate the following from English to Spanish.\
                               Provide only the translated  text:'{english_st}'""")
prompt=E_to_S_temp.invoke("Ai is great")
print(llm.invoke(prompt))