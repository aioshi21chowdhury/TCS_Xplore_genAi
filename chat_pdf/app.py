import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 

from langchain_gogole_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai