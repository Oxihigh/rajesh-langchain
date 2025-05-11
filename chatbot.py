import streamlit as st
import cohere
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain.llms import Cohere as CohereLLM
from langchain.embeddings.base import Embeddings

st.set_page_config(
    page_title="Rotaract Chat Bot",
    page_icon="🔴",
    layout="centered"
)
load_dotenv()
COHERE_API_KEY = os.getenv("API_KEY")
co = cohere.Client(COHERE_API_KEY)

class CustomCohereEmbeddings(Embeddings):
    def embed_documents(self, texts):
        response = co.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        return response.embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

embedding = CustomCohereEmbeddings()

llm = CohereLLM(cohere_api_key=COHERE_API_KEY, model="command-r-plus")

st.title("Ask me about rotaract")

file_path = "rotaract.txt"

try:
    with open(file_path, "r") as file:
        file_content = file.read()

    loader = TextLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embedding)
    retriever = vectorstore.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    for q, a in st.session_state.history:
        st.write(f"**You:** {q}")
        st.write(f"**AI:** {a}")

    user_input = st.text_input("Ask something:")

    if user_input:
        answer = qa.run(user_input)
        st.session_state.history.append((user_input, answer))
        st.session_state.user_input = ""
        st.write(f"**You:** {user_input}")
        st.write(f"**AI:** {answer}")

except FileNotFoundError:
    st.write(f"File {file_path} not found. Please place the file in the correct location.")
