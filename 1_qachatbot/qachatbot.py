import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Page config
st.set_page_config(page_title="Q&A Chatbot", page_icon="🤖", layout="wide")

# Title
st.title("Q&A Chatbot")
st.markdown("This is a simple Q&A chatbot built using Streamlit and LangChain. You can ask questions and get answers from the model.")


with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Select Model",
        ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 50, 1000, 200)
    load_dotenv()
    default_key = os.getenv("GROQ_API_KEY") or os.getenv("Groq_api_key") or ""
    api_key = st.text_input("Groq API Key", value=default_key, type="password", help="Enter your Groq API key. You can get it from https://console.groq.com/keys")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Initialize LLM
@st.cache_resource
def get_chain(api_key, model_name, temperature, max_tokens):
    if not api_key:
        return None
    llm = ChatGroq(api_key=api_key, model=model_name,
                   temperature=temperature, max_tokens=max_tokens, streaming=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{question}"),
    ])

    # Create chain
    chain = prompt | llm | StrOutputParser()
    return chain


# Get chain
chain = get_chain(api_key, model_name, temperature, max_tokens)

if not chain:
    st.warning("Please enter your Groq API key in the sidebar.")
    st.markdown("You can get your API key from [Groq](https://console.groq.com/keys).")

else:
    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if question := st.chat_input("Ask me anything"):
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Stream response from Groq
                for chunk in chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error: {str(e)}")
