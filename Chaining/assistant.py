from langchain.memory import VectorStoreRetrieverMemory #type: ignore
from langchain.vectorstores import FAISS #type: ignore
from langchain.embeddings import SentenceTransformerEmbeddings #type: ignore
from langchain.chains import ConversationalRetrievalChain #type: ignore
from langchain_groq import ChatGroq  #type: ignore
from langchain.memory import ConversationBufferMemory #type: ignore
from langchain.prompts import PromptTemplate #type: ignore

import os
from dotenv import load_dotenv  #type: ignore
load_dotenv()

vector_store_path = "../VectorStore/faiss_index"
GROQ_API_KEY = os.getenv("GROK_API_KEY")

# load vector store
def load_vector_store(path, embedding_model):
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)

# prompt
prompt_template = """
You are a personal assistant answering questions based only on the provided research papers. 
Use the retrieved document chunks as your sole source of information. 
Do not speculate, invent details, or use knowledge beyond the documents. 
If the documents lack sufficient information, respond with: 
"I don’t have enough information from the research papers to answer this."

Document context: {context}
Conversation history: {chat_history}
User question: {question}

Answer:
"""
QA_PROMPT =PromptTemplate(
    template=prompt_template,
    input_variables = ["context", "chat_history", "question"]
)


def initialize_assistant():
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = load_vector_store(vector_store_path, embedding_model)

    llm = ChatGroq(
        model="Gemma2-9b-It", 
        groq_api_key=GROQ_API_KEY
    )

    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        return_messages = True,
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k":3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents = True
    )

    return qa_chain

def ask_question(qa_chain, query):
    response = qa_chain.invoke({"question": query})
    return response["answer"]

if __name__ == "__main__":
    print("Initializing Personal Assistant...")
    qa_chain = initialize_assistant()
    print("Assistant is ready! Ask me anything about the research papers.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer = ask_question(qa_chain, user_input)
        print(f"Assistant: {answer}")








