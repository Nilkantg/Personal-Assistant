from langchain.document_loaders import PyPDFLoader #type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter #type: ignore
import pickle
import os

docs_folder = "docs"
output_file = "split_documents.pkl"

def load_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Loading {filename}...")

            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return documents
    
loaded_docs = load_documents(docs_folder)

# for doc in loaded_docs:
#     print(f"Page Content: {doc.page_content[:150]}")
#     print(f"Metadata: {doc.metadata}")
#     print("-" * 50)
    
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

split_docs = text_splitter.split_documents(loaded_docs)
print(f"split {len(loaded_docs)} pages into {len(split_docs)} chunks")

with open(output_file, "wb") as f:
    pickle.dump(split_docs, f)

print(f"saved loaded documents to {output_file}")
