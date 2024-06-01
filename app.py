import streamlit as st
from langchain_groq import ChatGroq
from langchain.document_loaders import YoutubeLoader, TextLoader, WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.summarize import load_summarize_chain
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
import pandas as pd
import os
import random
import string
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('groq_api_key')

models_in_Groq = ['gemma-7b-it', 'mixtral-8x7b-32768', 'llama3-8b-8192', 'llama3-70b-8192']

# Streamlit app layout
st.title("Multi-Purpose Query Application")

# Model selection
selected_model = st.selectbox("Select a model:", models_in_Groq)

llm = ChatGroq(temperature=0.7, groq_api_key=groq_api_key, model_name=selected_model)

# Functions for different functionalities
def summarize_youtube_video(youtube_url):
    loader = YoutubeLoader.from_youtube_url(youtube_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    texts = text_splitter.split_documents(transcript)
    chain = load_summarize_chain(llm=llm, memory=ConversationBufferMemory(memory_key="chat_history"), chain_type="map_reduce", verbose=True)
    summary = chain.run(texts)
    return summary

def run_retrieval_qa_text(url, query):
    loader = TextLoader(url)
    loaded_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(loaded_documents)
    embeddings = HuggingFaceEmbeddings()
    vector_store = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    result = qa.run(query)
    return result

def run_retrieval_qa_pdf(url, query):
    loader = PyPDFLoader(url)
    loaded_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(loaded_documents)
    embeddings = HuggingFaceEmbeddings()
    vector_store = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    result = qa.run(query)
    return result

def run_sql_query(file, query):
    df = pd.read_csv(file)
    random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=7))
    url = f"sqlite:///{random_name}.db"
    engine = create_engine(url)
    df.to_sql('rohan', engine, index=False)
    db = SQLDatabase(engine=engine)
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent_executor = create_sql_agent(llm, db=db, memory=memory, agent_type="openai-tools", verbose=True)
    response = agent_executor.invoke({"input": query})
    return response

def run_retrieval_qa_web(url, query):
    loader = WebBaseLoader(url)
    loaded_documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(loaded_documents)
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    result = qa.run(query)
    return result

option = st.selectbox(
    "Select an option:",
    ("Summarize YouTube Video", "Query Text File", "Query PDF File", "Query SQL Database", "Query Website")
)

if option == "Summarize YouTube Video":
    youtube_url = st.text_input("Enter YouTube URL:")
    if st.button("Summarize"):
        if youtube_url:
            summary = summarize_youtube_video(youtube_url)
            st.write("Summary:", summary)
        else:
            st.write("Please enter a valid YouTube URL")

elif option == "Query Text File":
    uploaded_file = st.file_uploader("Upload a text file", type="txt")
    query = st.text_input("Enter your query:")
    if st.button("Query"):
        if uploaded_file and query:
            with open("temp.txt", "wb") as f:
                f.write(uploaded_file.getbuffer())
            result = run_retrieval_qa_text("temp.txt", query)
            st.write("Result:", result)
        else:
            st.write("Please upload a text file and enter a query")

elif option == "Query PDF File":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    query = st.text_input("Enter your query:")
    if st.button("Query"):
        if uploaded_file and query:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            result = run_retrieval_qa_pdf("temp.pdf", query)
            st.write("Result:", result)
        else:
            st.write("Please upload a PDF file and enter a query")

elif option == "Query SQL Database":
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    query = st.text_input("Enter your query:")
    if st.button("Query"):
        if uploaded_file and query:
            with open("temp.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            result = run_sql_query("temp.csv", query)
            st.write("Result:", result)
        else:
            st.write("Please upload a CSV file and enter a query")

elif option == "Query Website":
    url = st.text_input("Enter website URL:")
    query = st.text_input("Enter your query:")
    if st.button("Query"):
        if url and query:
            result = run_retrieval_qa_web(url, query)
            st.write("Result:", result)
        else:
            st.write("Please enter a website URL and a query")
