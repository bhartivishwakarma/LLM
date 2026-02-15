from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
import re
import contractions


llm = OllamaLLM(
    model="gemma:2b",
    base_url="http://localhost:11434",
    temperature=0,
    streaming=False
)

parser = StrOutputParser()


Reviews = [
    "yeh product acha hai lekin thoda mehenga hai",
    "I LOVE this product!. It's absolutely amazing   ",
    "Terrible experience... I'm never buying again!!!"
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
    English Translation:
    """
)

translation_chain = translation_prompt | llm | parser

grammar_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Correct the grammar of the following English text:

    {text}
    Corrected Text:
    """
)

grammar_chain = grammar_prompt | llm | parser


sequential_chain = translation_chain | grammar_chain


sentiment_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    You are a sentiment analysis expert.

    Classify the sentiment as:
    Positive, Negative, or Neutral.

    Text: {text}
    Sentiment:
    """
)

sentiment_chain = sentiment_prompt | llm | parser


summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Give a short one-line summary of the following text:

    {text}
    Summary:
    """
)

summary_chain = summary_prompt | llm | parser


parallel_chain = RunnableParallel(
    sentiment=sentiment_chain,
    summary=summary_chain
)


for review in normalized_reviews:

    # Sequential Step
    corrected_text = sequential_chain.invoke({"text": review})

    # Parallel Step
    final_result = parallel_chain.invoke({"text": corrected_text})

    print("Original :", review)
    print("Corrected English :", corrected_text)
    print("Sentiment :", final_result["sentiment"])
    print("Summary :", final_result["summary"])
    print("-" * 60)
