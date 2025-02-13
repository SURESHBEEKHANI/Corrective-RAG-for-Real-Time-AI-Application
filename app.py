import streamlit as st
import requests

# Configure the Streamlit page
st.set_page_config(
    page_title="Corrective-RAG",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Page title and caption
st.title("⚡ Corrective-RAG Assistant")
st.caption("🤖 Smart AI Corrective-RAG Assistant")

# Custom CSS for styling (if needed)
# st.markdown("""
# <style>
# /* Add custom CSS here */
# </style>
# """, unsafe_allow_html=True)

# Sidebar with features and footer
with st.sidebar:
    st.divider()
    st.markdown("### Key Features")
    st.markdown("""
    - 🔎 Refinement
    - 🔄 Query Rewriting
    - � Accuracy
    - 📖 Self-Assessment
    - 🔎 Knowledge Refinement
    - 🔄 Query Rewriting & Web Search
    - � Improved Accuracy
    """)
    st.divider()
    st.markdown("**Built with** 🦙 Llama | LangGraph")

# FastAPI backend URL
API_URL = "http://localhost:8000/generate"  # Update this if hosted elsewhere

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{
        "role": "assistant",
        "content": "Hello! How can I assist you with your legal research today?"
    }]

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
question = st.chat_input("Enter your question:")

# Automatically generate an answer when a question is provided
if question and question.strip():  # Check if the input is not empty
    # Add user's question to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    # Display user's question in the chat
    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant's response
    with st.spinner("Generating answer..."):
        try:
            # Send request to the FastAPI backend
            response = requests.post(API_URL, json={"question": question})
            if response.status_code == 200:
                answer = response.json().get("answer", "No response generated.")
                # Add assistant's response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })
                # Display assistant's response in the chat
                with st.chat_message("assistant"):
                    st.markdown(answer)
            else:
                st.error(f"Error: Unable to connect to the API. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the API: {e}")