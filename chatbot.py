from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
import os
from rag import build_or_load_vector_store, create_rag_chain
import logging

# load environment variables from .env file
load_dotenv()
username = os.getenv("USER_NAME", "User")

@st.cache_resource
def get_rag_chain():
    vector_store = build_or_load_vector_store()
    llm = ChatGroq(
        model="openai/gpt-oss-safeguard-20b",
        temperature=0.01,
    )
    return create_rag_chain(vector_store, llm)

# streamlit app
st.set_page_config(
    page_title=f"{username}'s Chatbot",
    page_icon="🤖",
    layout="wide")
st.title(f"👾 {username}'s Chatbot 👾")
st.write(f"A friendly personal assistant for questions about {username}, her interests, and the things she enjoys exploring.")

# initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# show chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_prompt = st.chat_input("Ask me Anything!")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt}
    )

    try:
        with st.spinner("Searching the PDF knowledge base..."):
            rag_chain = get_rag_chain()
            result = rag_chain.invoke({"input": user_prompt})

    except FileNotFoundError as error:
        st.error(str(error))

    except Exception:
        logging.exception("Error during RAG chain invocation")
        st.error(
            "The assistant couldn't search the documents or generate a reply. "
            "Please try again; if the issue continues, check the app logs."
        )

    else:
        assistant_response = result["answer"]
        source_documents = result["context"]

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_response,
        })

        with st.chat_message("assistant"):
            st.markdown(assistant_response)
            st.markdown("**Sources:**")
            for doc in source_documents:
                st.markdown(
                    f"- {doc.metadata.get('source', 'Unknown source')} "
                    f"(page {doc.metadata.get('page_number', 'N/A')})"
                )