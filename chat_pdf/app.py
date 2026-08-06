# from pathlib import Path
# import os
# import streamlit as st
# import google.generativeai as genai
# from dotenv import load_dotenv

# # Strict compliance with your requested module list
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from pypdf import PdfReader

# # Path to the directory containing this script, then up one level to root
# root_dir = Path(__file__).resolve().parent.parent
# load_dotenv(dotenv_path=root_dir / ".env")

# api_key = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=api_key)


# # Extracting the pages from the pdf and returning the text
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"
#     return text


# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000, chunk_overlap=100
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")


# def generate_llm_response(context_text, user_question):
#     # Pure LCEL prompt structure using langchain_core
#     prompt = ChatPromptTemplate.from_template(
#         "You are a helpful assistant that answers questions about the uploaded document.\n"
#         "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
#         "Context:\n{context}\n\n"
#         "Question:\n{input}\n\n"
#         "Answer:"
#     )
    
#     model = ChatGoogleGenerativeAI(
#         #model="gemini-1.5-flash", temperature=0.2, max_output_tokens=512
#         model="gemini-2.5-pro", temperature=0.2, max_output_tokens=512
#     )

#     # Chain composed purely via the standard pipeline operator (|)
#     chain = prompt | model | StrOutputParser()
    
#     # Execute the pipeline directly
#     response = chain.invoke({"context": context_text, "input": user_question})
#     return response


# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

#     # Allowing dangerous deserialization since we generated the local index ourselves
#     new_db = FAISS.load_local(
#         "faiss_index", embeddings, allow_dangerous_deserialization=True
#     )
    
#     # Retrieve matching documents from local store
#     docs = new_db.similarity_search(user_question)
    
#     # Combine extracted page text manually to feed your layout smoothly
#     combined_context = "\n\n".join([doc.page_content for doc in docs])
    
#     # Process pipeline execution
#     response = generate_llm_response(combined_context, user_question)

#     st.write("Reply:", response)


# def main():
#     st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
#     st.header("Chat with PDF using Gemini 📚")

#     user_question = st.text_input("Ask a Question from the Uploaded PDFs:")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.subheader("Your Documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDF files and click on the Submit & Process Button",
#             accept_multiple_files=True,
#         )
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")


# if __name__ == "__main__":
#     main()
from pathlib import Path
import os
import json
import hashlib
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# LangChain imports (strictly as you requested)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Path to the directory containing this script, then up one level to root
root_dir = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=root_dir / ".env")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=api_key)

# ============ Configuration ============
INDEX_DIR = "faiss_index"
INDEX_META_FILE = "faiss_index_meta.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# LLM_MODEL = "gemini-1.5-flash"  # cheaper/faster than gemini-2.5-pro
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 512
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 3  # number of chunks to retrieve per query
# Gemini model for CHAT ONLY (not embeddings)
# Try in this order until one works:
# 1) "gemini-1.5-flash-latest"
# 2) "gemini-2.0-flash"
# 3) "gemini-2.5-flash"
LLM_MODEL = "gemini-1.5-flash-latest"
# =======================================

def files_hash(pdf_docs):
    """
    Compute a hash over uploaded PDF files to detect changes.
    pdf_docs: list of Streamlit UploadedFile objects
    """
    h = hashlib.md5()
    for pdf in pdf_docs:
        pdf.seek(0)
        h.update(pdf.read())
        pdf.seek(0)  # reset for later use
    return h.hexdigest()

def load_index_meta():
    """Load metadata about the current FAISS index (if it exists)."""
    if not Path(INDEX_META_FILE).exists():
        return None
    with open(INDEX_META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_index_meta(meta):
    """Save metadata about the current FAISS index."""
    with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_text_chunks(text):
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_text(text)
    return chunks

def build_vector_store(text_chunks):
    """
    Build a FAISS vector store using a LOCAL embedding model
    (no Google API calls for embeddings).
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(INDEX_DIR)
    return vector_store

def load_vector_store():
    """Load existing FAISS index from disk."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

def generate_llm_response(context_text, user_question):
    """
    Generate an LLM response using Google Gemini.
    Uses a cheaper/faster model and limited tokens.
    """
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant that answers questions about the uploaded document.\n"
        "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{input}\n\n"
        "Answer:"
    )
    
    model = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS
    )

    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"context": context_text, "input": user_question})
    return response

def user_input(user_question):
    """
    Retrieve relevant chunks from FAISS and generate an answer.
    No embedding API calls here; only the LLM uses Google.
    """
    if not Path(INDEX_DIR).exists():
        st.error("No index found. Please upload PDFs and click 'Submit & Process' first.")
        return
    
    vector_store = load_vector_store()
    docs = vector_store.similarity_search(user_question, k=TOP_K)
    
    if not docs:
        st.warning("No relevant context found for your question.")
        return
    
    combined_context = "\n\n".join([doc.page_content for doc in docs])
    response = generate_llm_response(combined_context, user_question)
    
    st.write("**Reply:**", response)

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
    st.header("Chat with PDF using Gemini 📚")

    user_question = st.text_input("Ask a Question from the Uploaded PDFs:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click on the Submit & Process Button",
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please select at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    # Compute hash of uploaded files
                    current_hash = files_hash(pdf_docs)
                    
                    # Check if we already have an index for this exact set of files
                    meta = load_index_meta()
                    if meta and meta.get("files_hash") == current_hash:
                        st.info("Index already exists for these files. Skipping re-embedding.")
                    else:
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("No text could be extracted from the PDFs.")
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            build_vector_store(text_chunks)
                            # Save metadata about this index
                            save_index_meta({
                                "files_hash": current_hash,
                                "num_chunks": len(text_chunks)
                            })
                            st.success("Index created/updated successfully.")
                
                st.success("Done")

if __name__ == "__main__":
    main()