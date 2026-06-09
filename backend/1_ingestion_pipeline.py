import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings for chat gpt api key
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """load all text file from the docs directory"""
    print(f"loading documents from {docs_path}")

    # Check if docs directory exist
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exitst. Please Create it.")

    loader = DirectoryLoader(
        path= docs_path,
        glob= "*.txt",
        loader_cls= TextLoader
    )

    documents= loader.load()
    if len(documents) == 0:
        raise FileNotFoundError(f"No text file found in {docs_path}. Please add your documents")
    
    for i, doc in enumerate(documents[:2]): #show the first two documents
        print(f"\nDocument{i+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} character")
        print(f" Content Preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=1500, chunk_overlap=200):
    """Split the documents into smaller chunks with overlap"""
    print("Splitting documents into chunks")
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} Character")
            print(f"Content:")
            print(chunk.page_content)
            print("_"*50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist Chromadb vector store"""
    print("Creating embedding and storing in ChromaDB...")

    # embedding_modal = OpenAIEmbeddings(modal="text-embedding-3-small")
    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Create ChromaDB vector store
    print("--- creating vector store ---")
    print(f"Total chunks: {len(chunks)}")
    vectorstore = Chroma.from_documents(
        documents = chunks, 
        embedding = embedding_model,
        persist_directory= persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    print(f"vector store create and store to {persist_directory}")
    return vectorstore

def main():
    print("Main function")
    # 1. Loading the files
    documents= load_documents(docs_path="docs")
    # 2. Chunking the files
    chunks = split_documents(documents)
    # 3. Embedding and storing in vector db
    create_vector_store(chunks)

if __name__ == "__main__":
    main()