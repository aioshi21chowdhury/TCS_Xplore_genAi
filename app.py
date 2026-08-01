# lets work on text  first

from dotenv import load_dotenv
load_dotenv() ## to load all the env variables


import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## function to loead models anf get responses
model=genai.GenerativeModel("gemini-3.5-flash-lite")
def get_gemini_response(question):
    response=model.generate_content(question)
    return response.text

## initiate the streamlit

st.set_page_config(page_title="Q&A Demo")

st.header("gemini LLM application")

input=st.text_input("Input: ",key="input")
submit=st.bottom.button("Ask the question")
if submit:
    response=get_gemini_response(input)
    st.subheader("the response of gemini model is: ")
    st.write(response)