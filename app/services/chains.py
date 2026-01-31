from typing import Literal, List

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings

# --- LLM Setup ---
# Use a cheaper model for routing, multi-query and grading to save costs.
# Use a stronger model for the actual answer generation.
small_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=settings.GROQ_API_KEY,
)
big_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=settings.GROQ_API_KEY
)

# --- 1. ROUTER CHAIN ---
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    # Added "generate" to the list of allowed values
    datasource: Literal["web_search", "vectorstore", "generate"] = Field(
        ...,
        description="Given a user question choose to route it to web_search, vectorstore, or generate."
    )


router_system = """
You are an expert at routing user queries to the appropriate data source.
Your task is to choose one of three paths based on the user's intent:

1. 'vectorstore': Use this for questions regarding the content of uploaded documents (PDFs, files) or specific domain knowledge provided by the user.
2. 'web_search': Use this ONLY when the user asks about real-world facts, current events, or information that is clearly NOT present in the uploaded documents or conversation history (e.g., "Current weather," "Stock prices," "Latest news").
3. 'generate': Use this for:
   - Greetings or general pleasantries.
   - Questions DIRECTLY related to the history of this current conversation (e.g., "What did we just talk about?", "Summarize our chat so far", "Who are you?", "Repeat my last question").
   - General creative or logic tasks (coding, writing, math) that do not require external context.

CRITICAL: If the user asks about the content of your previous messages or the conversation flow, you MUST choose 'generate'.
"""

router_prompt = ChatPromptTemplate.from_messages([
    ("system", router_system),
    ("human", "{question}"),
])

# Create the router chain with structured output
router_chain = router_prompt | small_llm.with_structured_output(RouteQuery)


# --- 2. MULTI-QUERY CHAIN (Query Translation) ---
class MultiQuery(BaseModel):
    """Generate multiple versions of a user question."""
    questions: List[str] = Field(
        ...,
        description="List of 3 alternative versions of the user question to improve retrieval coverage."
    )

multi_query_system = """
You are a retrieval optimizer. Generate 3 search queries to find the answer in a vector database.
Focus on:
1. Identifying key entities and technical terms.
2. Rephrasing the question as a statement that might appear in a document.
3. Breaking down complex questions into sub-parts.

Output ONLY the questions, one per line.
"""

multi_query_prompt = ChatPromptTemplate.from_messages([
    ("system", multi_query_system),
    ("human", "Question: {question}"),
])

# Create the multi-query chain with structured output
multi_query_chain = multi_query_prompt | small_llm.with_structured_output(MultiQuery)


# --- 3. DOCUMENT GRADER CHAIN (Are documents related?) ---
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

document_grader_system = """
You are an expert grader. Your task is to determine if a document chunk is USEFUL for answering the question.
- If the document contains any information, facts, or context that could contribute to the answer, grade it as 'yes'.
- Do not be overly strict. If there is semantic overlap, it's a 'yes'.
- Only grade 'no' if the document is completely irrelevant to the topic.
"""

document_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", document_grader_system),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

# Create the retrieval grading chain with structured output
document_grader_chain = document_grader_prompt | small_llm.with_structured_output(GradeDocuments)


# --- 4. GENERATION CHAIN ---
rag_system = """
You are a helpful AI assistant. Your primary goal is to answer the user's question using the provided Context.

GUIDELINES:
1. If the Context contains the answer, prioritize it and cite specific details.
2. If the Context is insufficient but related, use it as a base and supplement with your knowledge, but clearly state what comes from the context.
3. If the Context is truly empty or irrelevant, answer based on your internal knowledge.
4. Maintain a professional and concise tone.

CONTEXT FOR THE TASK:
{context}
"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

# Create the generation chain
generate_chain = rag_prompt | big_llm | StrOutputParser()


# --- 5. HALLUCINATION GRADER CHAIN ---
class GradeHallucinations(BaseModel):
    """Binary score for hallucination check on generated answer."""

    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


hallucination_system = """
You are a fact-checker. You must verify if the generated answer is supported by the 'Set of Facts'.
- Grade 'yes' if the answer's main claims are present in or logically follow from the facts.
- Grade 'no' only if the answer contradicts the facts or invents information not present in the provided context.
Ignore stylistic differences; focus on factual grounding.
"""

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", hallucination_system),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
])

# Create the hallucination grading chain with structured output
hallucination_chain = hallucination_prompt | small_llm.with_structured_output(GradeHallucinations)