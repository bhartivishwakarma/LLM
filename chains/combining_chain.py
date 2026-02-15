from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import re
import contractions


llm = OllamaLLM(
    model="gemma:2b",
    base_url="http://localhost:11434",
    temperature=0
)

parser = StrOutputParser()

Reviews = [
    "yeh product acha hai lekin thoda mehenga hai",
    "Pretty good, isn't it ? will buy again!"
]


def normalize_text(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

normalizer = RunnableLambda(lambda Reviews: [normalize_text(r) for r in Reviews])
normalized_reviews = normalizer.invoke(Reviews)


translation_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Translate the following text to English:

    {text}
    """
)


grammar_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Correct the grammar of the following English text:

    {text}
    """
)


sequential_chain = translation_prompt | llm | parser | grammar_prompt | llm | parser

for review in normalized_reviews:
    result = sequential_chain.invoke({"text": review})
    print("Original:", review)
    print("Final Output:", result)
    print("-" * 50)
