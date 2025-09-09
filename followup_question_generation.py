from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
llm=OllamaLLM(model="gemma:2b", temperature=0.1, max_tokens=1000)
statements=["I had a fantastic time hiking in the mountains yesterday.",
            "The new restaurant in town serves delicious vegetarian food.",
            "I am feeling quite stressed about the upcoming deadlines.",
            "Watching the sunset by the beach was a Calming experience.",
            "I recently started learning to play the guitar, and it's not so much fun as it's a initial stage!"]
followup_prompt=ChatPromptTemplate.from_template('''What is the appropriate and interesting followup Question for each
                                                  statements that i can learn more
                                                 about the provided text? only supply clear answers: {text}''')
f_prompt=followup_prompt.invoke({"text":statements})
print(llm.invoke(f_prompt))