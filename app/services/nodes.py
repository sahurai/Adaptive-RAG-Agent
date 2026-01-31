from langchain_core.messages import AIMessage
from langchain_tavily import TavilySearch
from app.core.config import settings
from app.services.chains import (
    router_chain,
    multi_query_chain,
    generate_chain,
    hallucination_chain, document_grader_chain
)
from app.services.db import get_retriever
from app.services.state import AgentState
from app.utils.rag_fusion import reciprocal_rank_fusion

# Initialize Tavily
web_search_tool = TavilySearch(k=3, tavily_api_key=settings.TAVILY_API_KEY)

# --- NODES ---

def router_node(state: AgentState):
    """Decides the initial path."""
    print("--- NODE: Router ---")

    source = router_chain.invoke({
        "question": state.question,
        "messages": state.messages
    })

    return {
        "route": source.datasource,
        "loop_step": 0,
        "documents": []
    }


def retrieve_node(state: AgentState):
    """Generates multiple queries, retrieves documents for each, and fuses results."""
    print("--- NODE: Retrieval (Multi-Query + Fusion) ---")
    question = state.question
    session_id = state.session_id
    retriever = get_retriever(session_id)

    # 1. Generate alternative questions
    multi_queries = multi_query_chain.invoke({"question": question})
    # Use original question + 3 generated ones
    queries = [question] + multi_queries.questions
    print(f"--- Generated {len(queries)} queries ---")

    # 2. Retrieve documents for each query
    final_results = []
    for q in queries:
        docs = retriever.invoke(q)
        final_results.append(docs)

    # 3. Apply RAG Fusion
    unique_docs = reciprocal_rank_fusion(final_results)

    return {"documents": [doc.page_content for doc in unique_docs]}


def grade_documents_node(state: AgentState):
    """Filters retrieved documents for relevance."""
    print("--- NODE: Grade Documents ---")
    question = state.question
    documents = state.documents

    filtered_docs = []
    for doc in documents:
        score = document_grader_chain.invoke({"question": question, "document": doc})
        if score.binary_score == "yes":
            filtered_docs.append(doc)
        else:
            continue

    print(f"--- GRADE: kept {len(filtered_docs)} relevant documents ---")
    return {"documents": filtered_docs}


def web_search_node(state: AgentState):
    """
    Minimalist web search node.
    Safely extracts content and ignores everything else.
    """
    print("--- NODE: Web Search ---")
    question = state.question
    current_step = state.loop_step
    new_step = current_step + 1

    try:
        # 1. Invoke tool
        response = web_search_tool.invoke({"query": question})

        content_list = []

        # 2. Strict check: only process if response is a dict with 'results'
        if isinstance(response, dict) and "results" in response:
            for item in response["results"]:
                # 3. Critical safety check: ensure item is a dict before accessing keys
                if isinstance(item, dict):
                    text = item.get("content", "")
                    if text:
                        content_list.append(text)

        # 4. Join valid results or use fallback
        if content_list:
            final_content = "\n\n".join(content_list)
        else:
            final_content = "No relevant information found."

        return {"documents": [final_content], "loop_step": new_step}

    except Exception as e:
        print(f"Error: {e}")
        return {"documents": ["Web search failed."], "loop_step": new_step}


def generate_node(state: AgentState):
    """Generates the final answer."""
    print("--- NODE: Generate ---")
    question = state.question
    documents = state.documents
    messages = state.messages

    if not documents:
        context = "No external context provided."
    else:
        context = "\n\n".join(documents)

    # Invoke the chain (which includes message history in the prompt)
    generation = generate_chain.invoke({
        "context": context,
        "question": question,
        "messages": messages
    })

    return {
        "generation": generation,
        "messages": [AIMessage(content=generation)]
    }


def hallucination_check_node(state: AgentState):
    """Checks for hallucinations. Includes loop breaker."""
    print("--- NODE: Hallucination Check ---")
    documents = state.documents
    generation = state.generation

    # 1. Loop Protection: If we retried too many times, force success
    if state.loop_step > 3:
        print(f"--- LOOP DETECTED (Step {state.loop_step}): Force finishing ---")
        return {"hallucination_grade": "yes"}

    # 2. If no docs, skip check
    if not documents:
        return {"hallucination_grade": "yes"}

    score = hallucination_chain.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    print(f"--- HALLUCINATION CHECK: {grade} (Step {state.loop_step}) ---")
    return {"hallucination_grade": grade}