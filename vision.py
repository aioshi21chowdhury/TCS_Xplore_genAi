# Let's work on images now

from dotenv import load_dotenv
load_dotenv() ## To load all the env variables

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image # FIX: Added missing PIL library import for images

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load models and get responses
# FIX: Updated to a valid multimodal model name
model = genai.GenerativeModel("gemini-2.5-flash") 

# FIX: Added 'input_prompt' as an explicit argument to the function
def get_gemini_response(image, input_prompt):
    if input_prompt != "":
        response = model.generate_content([input_prompt, image])
    else:
        response = model.generate_content(image)
     
    return response.text    

## Initialize Streamlit App

st.set_page_config(page_title="Gemini Image Demo")

st.header("Gemini Image Application")

# FIX: Changed variable name to 'input_prompt' to avoid overwriting Python's built-in input() function
input_prompt = st.text_input("Input prompt: ", key="input")

upload_file = st.file_uploader("Choose an image... ", type=["jpg", "jpeg", "png"])
image = ""

if upload_file is not None:
    image = Image.open(upload_file)
    # FIX: Updated deprecated use_column_width to use_container_width
    st.image(image, caption="Uploaded image.", use_container_width=True)
    
# FIX: Fixed typo from 'st.buttum' to 'st.button'
submit = st.button("Tell me about the image")    

## If submitted
# FIX: Wrapped in an if-statement so it only runs when the button is clicked
if submit:
    if image != "":
        response = get_gemini_response(image, input_prompt)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.error("Please upload an image first!")
