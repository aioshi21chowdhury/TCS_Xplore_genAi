import os
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import PIL.Image
import google.generativeai as genai

# Path to the directory containing this script, then up one level to root
root_dir = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=root_dir / ".env")

api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# function to load the gemini pro model
model=genai.GenerativeModel("gemini-2.0-flash-lite")

def get_gemini_response(input_,image,prompt):
    response = model.generate_content([input_,image[0],prompt])
    return response.text 

# image to bytes 
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        # read the file in to bytes
        byte_date=uploaded_file.getvalue()
        
        image_parts=[
            {
                "mime_type":uploaded_file.type,
                "data":byte_date
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("no file uploaded")
        

st.set_page_config(page_title="Multi_lingual text extractor")

st.header("Multi_lingual text extractor")


input_=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Upload a file", type=["jpg", "jpeg", "png"])
image=""

# uploading and showing the image
if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

submit=st.button("Tell me about the image")    

input_prompt="""
 your arae an expert in understanding different languages. 
 we will upload an image you have to just identify the which languages are written in the images.
"""
# if submit button is clicked get 
if submit:
    image_date=input_image_setup(uploaded_file)
    response=get_gemini_response(input_prompt,image_date,input_)
    st.subheader("the response of gemini model is: ")
    st.write(response)