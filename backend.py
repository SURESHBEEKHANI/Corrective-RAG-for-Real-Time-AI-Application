from fastapi import FastAPI
from pydantic import BaseModel
from src.required_function import (
    retrieve, grade_documents, generate, transform_query, web_search, decide_to_generate
)
from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict
import logging

# Define state class using TypedDict to represent the workflow state
class State(TypedDict):
    question: str  
    generation: str  
    web_search: str  
    documents: List[str]

# Define input model for FastAPI with Pydantic
class QueryInput(BaseModel):
    question: str

# Create workflow graph using the StateGraph
graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("grade_documents", grade_documents)
graph.add_node("generate", generate)
graph.add_node("transform_query", transform_query)
graph.add_node("web_search_node", web_search)

# Build the workflow edges
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges(
    "grade_documents", decide_to_generate,
    {"transform_query": "transform_query", "generate": "generate"}
)
graph.add_edge("transform_query", "web_search_node")
graph.add_edge("web_search_node", "generate")
graph.add_edge("generate", END)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app and compile workflow
app = FastAPI()
workflow = graph.compile()

@app.post("/generate")
def generate_response(query: QueryInput):
    """
    Endpoint that processes a question through the workflow and returns the generated response.
    """
    inputs = {"question": query.question}
    final_generation = None
    
    # Execute the workflow and process each step
    for output in workflow.stream(inputs):
        for key, value in output.items():
            logging.info(f"Processed node: {key}")
            # Capture generation if present in this step
            if "generation" in value:
                final_generation = value["generation"]
    
    return {"answer": final_generation}

# To run the application:
# uvicorn backend:app --reload