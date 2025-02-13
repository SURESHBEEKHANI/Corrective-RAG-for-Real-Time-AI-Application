import os
import logging  # Import logging module
from src.retriever import create_retriever, rag_chain, get_retrieval_grade, question_rewriter_chain  
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    filename="app.log",  # Log file name
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)

# Set API Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Initialize components
retriever = create_retriever()
chain = rag_chain()
retrieval_grader = get_retrieval_grade()
question_rewriter = question_rewriter_chain()
web_search_tool = TavilySearchResults(k=3)

def retrieve(state):
    """Retrieve relevant documents based on the question."""
    question = state["question"]
    logging.info(f"Retrieving documents for question: {question}")

    try:
        documents = retriever.get_relevant_documents(question)
        logging.info(f"Retrieved {len(documents)} documents.")
        return {"documents": documents, "question": question}
    
    except Exception as e:
        logging.error(f"Error during retrieval: {str(e)}")
        raise

def grade_documents(state):
    """Filter relevant documents based on grading."""
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"

    logging.info(f"Grading {len(documents)} retrieved documents for question: {question}")

    try:
        for d in documents:
            try:
                score = retrieval_grader.invoke({"question": question, "document": d.page_content})
                grade = score.binary_score
            except Exception as e:
                # Log error and treat document as non-relevant
                logging.error(f"retrieval_grader error: {e}")
                grade = "no"
            
            if grade.lower() == "yes":
                logging.info("Document graded as relevant.")
                filtered_docs.append(d)
            else:
                logging.info("Document graded as NOT relevant.")
                web_search = "Yes"

        return {"documents": filtered_docs, "question": question, "web_search": web_search}
    
    except Exception as e:
        logging.error(f"Error during document grading: {str(e)}")
        raise

def generate(state):
    """Generate an answer based on relevant documents."""
    question = state["question"]
    documents = state["documents"]
    logging.info(f"Generating response for question: {question} with {len(documents)} documents.")

    try:
        generation = chain.invoke({"context": documents, "question": question})
        # Fallback to a default message if no generation is returned
        if not generation:
            generation = "No answer generated. Please refine your query."
        logging.info("Successfully generated response.")
        return {"documents": documents, "question": question, "generation": generation}
    
    except Exception as e:
        logging.error(f"Error during generation: {str(e)}")
        raise

def transform_query(state):
    """Reformulate the question to improve retrieval results."""
    question = state["question"]
    logging.info(f"Transforming query: {question}")

    try:
        better_question = question_rewriter.invoke({"question": question})
        logging.info(f"Transformed query: {better_question}")
        return {"documents": state["documents"], "question": better_question}
    
    except Exception as e:
        logging.error(f"Error during query transformation: {str(e)}")
        raise

def web_search(state):
    """Perform a web search when retrieved documents are insufficient."""
    question = state["question"]
    logging.info(f"Performing web search for question: {question}")

    try:
        docs = web_search_tool.invoke({"query": question})
        web_results_content = "\n".join(
            d["content"] if isinstance(d, dict) and "content" in d else d for d in docs
        )
        state["documents"].append(Document(page_content=web_results_content))

        logging.info(f"Web search retrieved {len(docs)} results.")
        return {"documents": state["documents"], "question": question}
    
    except Exception as e:
        logging.error(f"Error during web search: {str(e)}")
        raise

def decide_to_generate(state):
    """Determine whether to generate an answer or refine the question."""
    logging.info("Assessing graded documents for decision making.")

    if state["web_search"] == "Yes":
        logging.info("Decision: Transform query.")
        return "transform_query"
    else:
        logging.info("Decision: Generate response.")
        return "generate"
