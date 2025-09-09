from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
llm=OllamaLLM(model="gemma:2b", temperature=0.1, max_tokens=1000)

statements=["I had a fantastic time hiking in the mountains yesterday.",
            "The new restaurant in town serves delicious vegetarian food.",
            "I am feeling quite stressed about the upcoming deadlines.",
            "Watching the sunset by the beach was a Calming experience.",
            "I recently started learning to play the guitar, and it's not so much fun as it's a initial stage!"]
sentiment_prompt=ChatPromptTemplate.from_template('''In a single word Either Positive or negative,\
                                                  provide the sentiment of the following piece\
                                                   of text :{text}''')
s_prompt=sentiment_prompt.invoke({"text":statements})
print(llm.invoke(s_prompt))