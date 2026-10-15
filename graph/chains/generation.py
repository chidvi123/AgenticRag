from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

prompt = ChatPromptTemplate(
    [
        (
            "human",
            "You are an assistant for question-answering tasks ."
            "Use the following pieces of retrieved context to answer the question "
            "If you dont know the answer , just say that you dont know "
            "Use three sentnces maximum and keep the answer concise .\n"
            "Question : {question} \n context :{context} \n Answer :"
        )
    ]
)


generation_chain = prompt | llm | StrOutputParser()