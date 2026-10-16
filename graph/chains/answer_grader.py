from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from pydantic import BaseModel,Field

class GradeAnswer(BaseModel):
    binary_score : bool = Field(
        description = "Answer address the question 'yes' or 'no' "
    )

llm = ChatGroq(model="qwen/qwen3.8-27b",temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)


system ="""
    You are a grader assessing whether the answer addresses / resolves a question 
    Give a binary score 'yes' or 'no' , 'yes' means that the answer resolves the question . 
    """

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question:\n\n{question}\n\nLLM generation:\n\n{generation}"
        )
    ]
)

answer_grader_chain = answer_prompt | structured_llm_grader