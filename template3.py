from langchain_ollama import OllamaLLM
llm=OllamaLLM(model="gemma:2b", temperature=0.1, max_tokens=1000)
def translate(from_language,to_language,content):
    return f"Translate the following from {from_language}to{to_language}.Provide only translated text:{content}"
print(llm.invoke(translate("English","chinese","Hello")))