from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
llm=OllamaLLM(model="gemma:2b",temperature=0)

translate_prompt=ChatPromptTemplate.from_template("Translate the following from{from_language} to {to_language}.Provide only the translated text:{content}")
prompt=translate_prompt.invoke({"from_language":"English","to_language":"French","content":"I love programming"})
print(llm.invoke(prompt))