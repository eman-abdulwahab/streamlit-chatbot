from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq

# load environment variables from .env file
load_dotenv()

eman_background_info = """
Eman is a curious, adventurous soul who loves learning new things, exploring, and embracing spontaneity. She’s interested in technology, enjoys learning Spanish, and loves the idea of dancing and trying different physical activities. She’s always curious about how things work and enjoys discovering new interests. She’s also a proud mom to Sami (5) and Maryam (5 months).

When answering questions about Eman, respond in a natural, warm, conversational way that reflects her personality. Be curious, playful, and authentic rather than sounding like a formal biography. Never invent details about her.
"""

# streamlit app
st.set_page_config(
    page_title="Eman's Chatbot", 
    page_icon="🤖",
    layout="wide")
st.title("👾 Eman's Chatbot 👾")
st.write("A friendly personal assistant for questions about Eman, her interests, and the things she enjoys exploring.")

# initiat chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# show chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# initiate llm
llm = ChatGroq(
    model = "openai/gpt-oss-safeguard-20b",
    temperature = 0.1
)

user_prompt = st.chat_input("Ask me Anything!")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append(
        {"role":"user", "content": user_prompt}
        )

    response = llm.invoke(
        input=[
            {"role":"system", "content": "You are Eman's personal assistant. Use the following info to answer questions about her:\n"+ eman_background_info},
            *st.session_state.chat_history
            ]
    )

    assistant_response = response.content
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_response
    })

    with st.chat_message("assistant"):
        st.markdown(assistant_response)