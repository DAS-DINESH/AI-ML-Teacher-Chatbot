# app.py
import os
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
# Set page configuration
st.set_page_config(page_title="DeepSeek Chat App", page_icon="💬")

# App title and description
st.title("🤖 Ai/ML Teacher")
st.markdown("Ask questions and get answers related to ai")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    groq_api_key = os.getenv("GROQ_API")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app uses LangChain with Groq's DeepSeek model")
    st.markdown("Created with Streamlit 🎈")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to generate response
def generate_response(prompt, api_key):
    # Create LLM
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
    )
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_template(
        template=""" system: You are an Ai/ML teacher. if you are asked any question other than
        field of you AI/ML you will simply answer that you are not aware of this thing.
        And when you are answering give ouput in very good and easy way of explaination 
        about that particular topic.
        
        Human: {input}
        
        AI: """
    
    )
    
    # Create chain
    chain = RunnablePassthrough()|prompt_template|llm|StrOutputParser()
    
    # Run chain
    response = chain.invoke(prompt)
    return response

# Chat input
if prompt := st.chat_input("What is Linear Regression?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, groq_api_key)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
