from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from app.services.state import AgentState
from app.services.nodes import (
    router_node,
    retrieve_node,
    grade_documents_node,
    web_search_node,
    generate_node,
    hallucination_check_node
)

# 1. Initialize Graph
workflow = StateGraph(AgentState)

# 2. Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)
workflow.add_node("hallucination_check", hallucination_check_node)

# 3. Entry Point
workflow.set_entry_point("router")


# 4. Router Decision
def route_decision(state):
    if state.route == "web_search":
        return "web_search"
    elif state.route == "vectorstore":
        return "retrieve"
    else:
        return "generate"


workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "web_search": "web_search",
        "retrieve": "retrieve",
        "generate": "generate",
    }
)

# 5. Retrieve -> Grade
workflow.add_edge("retrieve", "grade_documents")


# 6. Grade Decision
def decide_to_generate(state):
    filtered_docs = state.documents
    if not filtered_docs:
        # No relevant docs -> fallback to web
        return "web_search"
    else:
        return "generate"


workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)

# 7. Standard Edges
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "hallucination_check")


# 8. Hallucination Decision (With Loop Break)
def hallucination_decision(state):
    # If we exceeded retry limit (3), assume useful to stop loop
    if state.loop_step > 3:
        return "useful"

    if state.route == "generate":
        return "useful"

    if state.hallucination_grade == "yes":
        return "useful"
    else:
        # Hallucination detected -> Retry search
        return "hallucinated"


workflow.add_conditional_edges(
    "hallucination_check",
    hallucination_decision,
    {
        "useful": END,
        "hallucinated": "web_search"
    }
)

# Initialize memory
memory = MemorySaver()

# Compile graph
app_graph = workflow.compile(checkpointer=memory)