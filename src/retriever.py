import os
import logging  # Import logging module
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

# Set up logging
logging.basicConfig(
    filename="app.log",  # Log file name
    level=logging.INFO,  # Set logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(model_name="llama-3.1-8b-instant")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_retriever():
    """Creates a retriever by loading and indexing documents."""
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    try:
        logging.info("Loading documents from URLs...")
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=20
        )
        doc_splits = text_splitter.split_documents(docs_list)

        logging.info(f"Successfully split {len(doc_splits)} document chunks.")

        vectorstore = Chroma.from_documents(
            documents=doc_splits, collection_name="rag-chroma", embedding=embeddings
        )

        logging.info("Document retrieval setup completed.")
        return vectorstore.as_retriever()

    except Exception as e:
        logging.error(f"Error in create_retriever: {str(e)}")
        raise

def rag_chain():
    """Creates a RAG chain for retrieval-augmented generation."""
    try:
        logging.info("Setting up RAG chain...")
        prompt = hub.pull("rlm/rag-prompt")
        return prompt | llm | StrOutputParser()
    
    except Exception as e:
        logging.error(f"Error in rag_chain: {str(e)}")
        raise

def get_retrieval_grade():
    """Creates a retrieval grader chain."""
    try:
        structured_llm_grader = llm.with_structured_output(GradeDocuments)

        system = """You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
        Give a binary score 'yes' or 'no'."""

        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])

        logging.info("Retrieval grading chain setup completed.")
        return grade_prompt | structured_llm_grader

    except Exception as e:
        logging.error(f"Error in get_retrieval_grade: {str(e)}")
        raise

def question_rewriter_chain():
    """Creates a question rewriter chain."""
    try:
        system_message = """You are a question re-writer that converts an input question to a better version 
        optimized for web search. Look at the input and try to reason about the underlying semantic intent."""

        re_write_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ])

        logging.info("Question rewriter chain setup completed.")
        return re_write_prompt | llm | StrOutputParser()

    except Exception as e:
        logging.error(f"Error in question_rewriter_chain: {str(e)}")
        raise
