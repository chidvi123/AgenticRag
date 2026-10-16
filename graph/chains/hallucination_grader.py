from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from pydantic import BaseModel,Field


llm = ChatGroq(model="qwen/qwen3.8-27b",temperature=0)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation asnwer """

    binary_score : bool = Field(
        description = "Answer is grounded in the facts , 'yes' or 'no' "
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)


system ="""
    You are a grader assessing whether an LLM generation is grounded in / Supported by a set of retrieved facts.
    Give a binary score 'yes' or 'no' , 'yes' means that the answer is grounded in / supported by set of facts 
"""

hallucination_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","set of facts :\n\n {documents} \n\n LLm generation :{generation}"),
    ]
)

hallucination_grader_chain = hallucination_prompt | structured_llm_grader