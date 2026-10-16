from dotenv import load_dotenv

from langgraph.graph import END,StateGraph

from graph.consts import RETRIEVE,GENERATE,GRADE_DOCUMENTS,WEBSEARCH
from graph.nodes  import generate,grade_documents,retrieve,web_search
from graph.state import GraphState
from graph.chains.answer_grader import answer_grader_chain
from graph.chains.hallucination_grader import hallucination_grader_chain

load_dotenv()



def decide_to_generate(state):
    print("---Assess GRADED DOCUMENTS ---")

    if state["web_search"]:
        print(
            "---DECISION : NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION "
            )
        return WEBSEARCH
    else:
        print("---DECISION GENERATE---")
        return GENERATE

def grade_generation_in_documents_and_question(state:GraphState) -> str:
    print("CHECKING HALLUCINATIONS")
    question = state['question']
    documents=state['documents']
    generation = state['generation']

    score = hallucination_grader_chain.invoke(
        {"documents":documents,"generation":generation}
    )

    if hallucination_grade := score.binary_score:
        print("--- DECISION : GENERATION IS GROUNDED IN DOCUMENTS ---")
        print("--- GRADE GENERATION VS QUESTION ---")
        score = answer_grader_chain.invoke({"question":question,"generation":generation})

        if answer_grade := score.binary_score:
            print("--- DECISION : GENERATION ADDRESS QUESTION ---")
            return 'useful'
        else:
            print("--- DECISION : GENERATION DOESNT ADDRESS THE QUESTION ---")
            return 'not useful'
    else:
        print("--- DECISION : GENERATION IS NOT GROUNDED IN DOCUMENTS ---")
        return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE,retrieve)
workflow.add_node(GRADE_DOCUMENTS,grade_documents)
workflow.add_node(GENERATE,generate)
workflow.add_node(WEBSEARCH,web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE,GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH:WEBSEARCH,
        GENERATE:GENERATE,
    },

)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_in_documents_and_question,
    {
        "not supported":GENERATE,
        "useful":END,
        "not useful":WEBSEARCH
    },
)

workflow.add_edge(WEBSEARCH,GENERATE)
workflow.add_edge(GENERATE,END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")