from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
llm=OllamaLLM(model="gemma:2b",temperature=0,max_tokens=1000)

def  translate(english_st):
    return f"Translate the folowing from English to Spanish.Provide just the Translated text:{english_st}"
english_st=["Toady is a good day","Tommorrow will be even better","Next week,who can say"]
prompts=[translate(english_st)]
translations=llm.batch(prompts)
for translation in translations:
    print(translation,end="",flush=True)