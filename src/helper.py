# from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings



def load_data(data):
    loader = PyPDFDirectoryLoader(data)
    documents = loader.load()
    return documents

def text_split(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    data_splitter = text_splitter.split_documents(data)
    return data_splitter

def embedd_model():
    embedding = HuggingFaceEmbeddings(
        model_name= "sentence-transformers/all-mpnet-base-v2"
    )
    return embedding


