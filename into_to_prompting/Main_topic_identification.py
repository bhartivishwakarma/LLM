from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
llm=OllamaLLM(model="gemma:2b", temperature=0.1, max_tokens=1000)
statements=["I had a fantastic time hiking in the mountains yesterday.",
            "The new restaurant in town serves delicious vegetarian food.",
            "I am feeling quite stressed about the upcoming deadlines.",
            "Watching the sunset by the beach was a Calming experience.",
            "I recently started learning to play the guitar, and it's not so much fun as it's a initial stage!"]
main_topic_prompt=ChatPromptTemplate.from_template('''Identify and state each statements as concisely as possible,the main
                                                   topic of the following piece of text.only provide the
                                                    main topic and no other helpful comments. text:{text}''')
main_prompt=main_topic_prompt.invoke({"text":statements})
print(llm.invoke(main_prompt))