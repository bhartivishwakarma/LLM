from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
llm=OllamaLLM(model="gemma:2b",base_url="http://localhost:11434",temperature=0,streaming=False)

parser = StrOutputParser()

Reviews=[" I LOVE this product!. It's absolutely amazing   ",
         "Not Bad,  but could be better. I've seen worse.",
         "Terrible experience... I'm never buying again!!!",
         "Pretty good, isn't it ? will buy again!"
         ]

#fuctions to normalize Reviews
import re
import contractions

def normalize_text(text):

    text= text.lower()
    text=contractions.fix(text)
    text=re.sub(r"\s+"," ",text).strip()
    return text
normalizer= RunnableLambda(lambda Reviews: [normalize_text(r) for r in Reviews])

normalized_reviews = normalizer.invoke(Reviews)
print(normalized_reviews)

#Sentiment Template

sentiment_template=PromptTemplate(input_variable=["text"],
                                  template="""you are a sentiment analysis expert.
                                  
                                  Classify the sentiment of the following review as : Positive, Negative,or Neutral.

                                  Text:{text}
                                  sentiment:
                                  """)

sentiment=RunnableLambda(lambda normalized_reviews :[{"text":i} for i in normalized_reviews])#convert list into dict format

chain= sentiment_template | llm | parser

for review in normalized_reviews:
    result= chain.invoke({"text":review})
    print("Review: ",review)
    print("Sentiment:",result)


#batch_result= chain.batch([{"text":r} for r in normalized_reviews])
#print(batch_result) (using batch)


