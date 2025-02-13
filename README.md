# Corrective-RAG-for-Real-Time-AI-Application

## Overview
Corrective-RAG-for-Real-Time-AI-Application is an advanced system that implements a corrective Retrieval-Augmented Generation (RAG) approach tailored for real-time AI applications. By combining intelligent document retrieval, query refinement, and LLM-powered response generation, the system delivers accurate and context-aware answers.

## Features
- High-quality document retrieval and relevance grading
- Intelligent answer generation with state-of-the-art LLMs
- Automated query rewriting for improved search results
- Integrated web search to supplement document retrieval
- backend with FastAPI and an interactive UI using Streamlit

## Installation
```bash
# Clone the repository
git clone https://github.com/SURESHBEEKHANI/Corrective-RAG-for-Real-Time-AI-Application.git
cd Corrective-RAG-for-Real-Time-AI-Application

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Start the API server
uvicorn backend:app --reload

# Run the Streamlit application
streamlit run app.py
```

## Project Structure
```
f:/
├── backend.py
├── app.py
├── src/
│   └── ... (retriever.py, required_function.py, ...)
├── resource/
│   └── corrective_rag.ipynb
├── requirements.txt
└── README.md
```

## Architecture
- **Retriever Module:** Fetches relevant documents from various sources.
- **RAG Chain:** Generates context-aware answers by combining retrieved documents with LLM outputs.
- **Query Rewriter:** Optimizes user queries, enhancing retrieval quality.
- **Web Search Integration:** Supplements document retrieval through real-time web search.
  

## Contributing
Contributions are welcome. Please fork the repository and create a pull request with your proposed enhancements or bug fixes.

## License
Distributed under the MIT License. See LICENSE for more information.
