from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource"""

    datasource : Literal["vectorstore","websearch"] = Field(
        description="Given a user question choose to route it to web search or a vectorstore."
    )


llm = ChatGroq(model="qwen/qwen3.8-27b",temperature=0)

structured_llm_router = llm.with_structured_output(RouteQuery)


system =""""
You are an expert IN routing a user question to a vector store or web search .
The vectorstore contains documents related to agents , prompt engineering , and adversarial attacks.
Use the vectorstore for questions on these topics . For all else , use web-search .
"""

route_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","{question}"),
    ]
)

question_router = route_prompt | structured_llm_router