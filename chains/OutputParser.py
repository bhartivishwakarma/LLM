from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
llm=OllamaLLM(model="gemma:2b",base_url="http://localhost:11434",temperature=0,streaming=False)

parser=StrOutputParser()
parser.invoke("parse this string")
parser.batch(["parse this string","parse that string"])
template=PromptTemplate.from_template("Answer the following Question:{question}")
chain=template | llm | parser
print(chain.get_graph().draw_ascii())
answer=chain.invoke({"question":"What is the capital of india?"})
print(answer)