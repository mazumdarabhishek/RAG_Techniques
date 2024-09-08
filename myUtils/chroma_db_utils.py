from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from tqdm import tqdm
import os


def load_documents_from_directory(data_path:str):
    """
    Function to load all pdf documents from specified directory
    :param data_path:  path
    :return: Langchain document object
    """
    loader = PyPDFDirectoryLoader(data_path, glob="*.pdf")
    documents = loader.load()
    return documents

def split_into_chunks(documents: list[Document], chunk_size=1000, chunk_overlap=300):
    """
    Function to split documents into chunks as per specified parameters
    :param documents: list of langchain document objects
    :param chunk_size: number of tokens per each chunk
    :param chunk_overlap: number of tokens to overlap between subsequent chunks
    :return: list of chunks as langchain document
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap= chunk_overlap,
        length_function=len,
        add_start_index=True
    )

    chunks = splitter.split_documents(documents)
    print(f"{len(documents)} were split into {len(chunks)}")
    return chunks

def load_data_into_chroma(chunks:list, embedding_function,
                          collection_name:str='sample_collection', to_persist:bool=False, persist_dir:str='Chroma'):
    """
    Function to load data to Chromadb
    :param chunks: langchain Document objects
    :param embedding_function: Embedding function to use for deriving vector representations of chunks
    :param collection_name: name of the collection
    :param to_persist: whether to persist the db to disk or not
    :param persist_dir: if to_persist, then a directory called 'Chroma' will be created
    :return: Chroma.as_retriever() object
    """

    if to_persist:
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_dir)
    else:
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function)

    for chunk in tqdm(chunks[:50]):
        db.add_documents([chunk])

    print("Data Loaded to Chroma Collection successfully")

    return db
