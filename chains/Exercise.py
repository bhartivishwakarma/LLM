#create a chain that is able to transalte a given statement source language and target language you specify
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
llm=OllamaLLM(model="gemma:2b",base_url="http://localhost:11434",temperature=0,streaming=False)

translate_template=PromptTemplate.from_template("""Translate the following statement \
                                                from{from_language} to {to_language} Provide only the \
                                                Translated text :{statement}""")
parser=StrOutputParser()
translation_chain=translate_template | llm | parser
answer=translation_chain.invoke({"from_language":"English","to_language":"German",\
                                 "statement":"No matter Who you are it's \
                                    fun to learn new things"}) 
print(answer)
