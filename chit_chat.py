from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## function to load Gemini model
model = genai.GenerativeModel("gemini-3.5-flash-lite")

## first Initialize chat session
if "chat" not in st.session_state:
    st.session_state["chat"] = model.start_chat(history=[])

def get_gemini_response(question):
    response = st.session_state["chat"].send_message(question, stream=True)
    return response

## Initialize Streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("Gemini LLM Application")

## Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

input_txt = st.text_input("Input:", key="input")
submit = st.button("Ask the Question")

if submit and input_txt:

    response = get_gemini_response(input_txt)

    ## Store user question
    st.session_state["chat_history"].append(("You", input_txt))

    st.subheader("Response")

    full_response = ""

    for chunk in response:
        st.write(chunk.text)
        full_response += chunk.text

    ## Store complete bot response
    st.session_state["chat_history"].append(("Bot", full_response))

st.subheader("Chat History")

for role, text in st.session_state["chat_history"]:
    st.write(f"{role}: {text}")