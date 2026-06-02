from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

def load_documents(data_folder="../data"):
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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store