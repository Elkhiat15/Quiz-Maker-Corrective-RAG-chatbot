from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List, Any
from agents.agents import (
    retrieve,
    grade_documents,
    generate,
    transform_query,
    web_search,
    decide_to_generate
)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: str
    retriever: Any
    documents: List[str]
    relevant_images: List[str]


# Create
def create_graph():
    graph = StateGraph(GraphState)

    # Define the nodes
    graph.add_node("retrieve", retrieve)  # retrieve
    graph.add_node("grade_documents", grade_documents)  # grade documents
    graph.add_node("generate", generate)  # generate
    graph.add_node("transform_query", transform_query)  # transform_query
    graph.add_node("web_search_node", web_search)  # web search

    # Build graph
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    graph.add_edge("transform_query", "web_search_node")
    graph.add_edge("web_search_node", "generate")
    graph.add_edge("generate", END)

    return graph


# Compile
def compile_workflow():
    graph = create_graph()
    workflow = graph.compile()
    return workflow
