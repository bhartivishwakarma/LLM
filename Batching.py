from langchain_ollama import OllamaLLM
llm=OllamaLLM(base_url="http://localhost:11434",model="gemma:2b",temperature=0)
state_capital_question=["What is the capital of California?"
                         "What is the capital of Texas?"
                          "What is the capital of New York?"
                          "What is the capital of Florida?"
                          "What is the capital of Illinois?"
                           "What is the capital of Ohio?"]

                            
capitals=llm.batch(state_capital_question)
for capital in capitals:
  print(capital,end=" ",flush=True)

