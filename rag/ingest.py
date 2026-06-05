from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from ragas.embeddings import embedding_factory
from openai import OpenAI, AsyncOpenAI
import os

# LangChain embeddings — for FAISS vectorstore
langchain_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Ragas embeddings — for evaluation only
sync_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
async_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# for ragas eval (async)
ragas_embeddings = embedding_factory('openai', model='text-embedding-ada-002', client=async_client, interface='modern')

def load_documents(data_folder="../data/context_corpus"):
    docs = []
    for pdf in Path(data_folder).glob("*.pdf"):
        loader = PyMuPDFLoader(str(pdf))
        docs.extend(loader.load())
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

def build_vectorstore(split_docs):
    vector_store = FAISS.from_documents(split_docs, langchain_embeddings)
    return vector_store

def get_embeddings():
    return ragas_embeddings