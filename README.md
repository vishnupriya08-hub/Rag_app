RAG Q&A App Documentation

Overview

The RAG Q&A App is a Retrieval-Augmented Generation (RAG) application built using Streamlit. It allows users to upload documents (PDF, DOCX, TXT), enter text, or provide links, and then query the extracted content. The system processes the input data, generates vector embeddings using FAISS and Hugging Face Transformers, and retrieves context-aware answers using a Meta-Llama-3-8B-Instruct model hosted on Hugging Face.

Installation

To run the application, ensure you have the following dependencies installed:

pip install streamlit faiss-cpu numpy langchain langchain-community langchain-huggingface PyPDF2 python-docx

Features

Multiple Input Types: Supports PDF, DOCX, TXT files, web links, and raw text.

Text Processing: Extracts content and splits it into manageable chunks.

Embedding Generation: Uses sentence-transformers/all-mpnet-base-v2 to convert text into embeddings.

Vector Search: Utilizes FAISS for efficient semantic search.

LLM Response Generation: Uses Meta-Llama-3-8B-Instruct for intelligent responses.

Interactive UI: Built with Streamlit for an intuitive experience.

Code Breakdown

1. Import Required Libraries

import streamlit as st
import faiss
import os
from io import BytesIO
from docx import Document
import numpy as np
from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEndpoint
from langchain.llms import HuggingFaceEndpoint

This section loads the required libraries for handling document parsing, vector embedding, and LLM-based response generation.

2. Set Up the Hugging Face Model

huggingface_api_key = "your_huggingface_api_key"  
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key  

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    token=huggingface_api_key,
    temperature=0.6,
    task="text-generation"
)

Defines the Meta-Llama-3-8B-Instruct LLM hosted on Hugging Face for query responses.

3. Process Input Data and Generate FAISS Index

def process_input(input_type, input_data):

This function handles different input formats, extracts text, splits it into smaller chunks, and generates embeddings.

Supports: Links, PDF, DOCX, TXT, and raw text.

Embeddings: Generated using sentence-transformers/all-mpnet-base-v2.

Vector Indexing: Uses FAISS for fast retrieval.

4. Query Answering with RAG

def answer_question(vectorstore, query):

This function retrieves relevant text chunks and generates an answer using the Meta-Llama-3-8B-Instruct model.

5. Streamlit UI Implementation

def main():

Allows users to upload files, enter text, or provide links.

Processes input data and stores it in session state.

Accepts a user query and generates an answer.

Usage

Run the Streamlit App

streamlit run app.py

Select Input Type (Link, PDF, Text, DOCX, TXT).

Upload the file or enter the text/link.

Click 'Proceed' to process the input.

Enter your question and submit.

View the generated answer from the AI model.

Future Enhancements

Add support for multi-file document retrieval.

Implement a more advanced ranking mechanism for retrieval.

Optimize vector storage for better performance.

Deploy the application on Hugging Face Spaces or AWS.

Conclusion

The RAG Q&A App is a powerful, interactive tool that combines document parsing, semantic search, and LLMs to answer queries contextually. It leverages FAISS for efficient retrieval and Meta-Llama-3-8B-Instruct for intelligent response generation. 
