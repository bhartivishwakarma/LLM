from langchain_ollama import ChatOllama
llm=ChatOllama(base_url="http://localhost:11434",model="gemma:2b",temperature=0,streaming=False)

faq_questions=["What is a Large Language Model?",
               "How do LLMS Work?",
               "How do LLMS generate text?"]
def create_faq_doc(faq_questions,faq_answers):
    faq_document=" "
    for question,response in zip(faq_questions,faq_answers):
        faq_document+=f'{question.upper()}\n\n'
        faq_document+=f'{response.content}\n\n'
        faq_document+="_"*30+"\n\n"

    return faq_document
faq_answers=llm.batch(faq_questions)
print(create_faq_doc(faq_questions,faq_answers))