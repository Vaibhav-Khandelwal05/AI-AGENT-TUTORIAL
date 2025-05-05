from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


model = OllamaLLM(model="llama3.2")

template = """

you are an expert in answering questions about a pizza restaurant.

here is the question to answer: {question}

here are some relevant reviews: {reviews}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n-----------------------")
    question = input("Ask your quesetion (press q to quit): ")
    print("\n\n")
    if question == "q":
        break

    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews":[reviews] , "question": question})
    print(result)