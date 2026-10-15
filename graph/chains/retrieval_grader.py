from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq

llm = ChatGroq(model="qwen/qwen3.8-27b",temperature=0)


class GradeDocuments(BaseModel):
    """Binary score relevance check in retrieved documents """

    binary_score : str = Field(
        description = "Documents are relevant to the question , 'yes' or 'no' "
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)


system = """
You are a grader assessing relevance of a retrieved document to a user question . \n
If the document contains keyword(s) or sematic meaning related to the question , grade it as relevance . \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question .
"""

grade_prompt=ChatPromptTemplate(
    [
        ("system",system),
        ("human","Retrieved document : \n\n {document} \n\n User question :{question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


