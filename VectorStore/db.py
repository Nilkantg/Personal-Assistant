from langchain.vectorstores import Chroma, FAISS #type: ignore
from langchain.embeddings import SentenceTransformerEmbeddings #type: ignore
from langchain.memory import VectorStoreRetrieverMemory #type: ignore
import os
import pickle
from dotenv import load_dotenv #type: ignore
load_dotenv()

# Initialize embeddings
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

documents_path = "D:\Gen_AI Projects\Personal-Assistant\Data_Loaders\split_documents.pkl"
vector_store_path = "faiss_index"

def load_split_documents(file_path):
    with open(documents_path, "rb") as f:
        split_docs = pickle.load(f)

    return split_docs

def create_vector_store(documents, emebdding_model):
    vectore_store = FAISS.from_documents(documents, embedding_model)
    
    return vectore_store

if __name__ == "__main__":
    # load documents
    split_docs = load_split_documents(documents_path)
    print(f"Loaded {len(split_docs)} document chunks from {documents_path}")

    vector_store = create_vector_store(split_docs, embedding_model)
    vector_store.save_local(vector_store_path)
    print(f"Vector store saved to {vector_store_path}")
















