import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)

def build_qa_chain(docs):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    pdf_path = input("Enter path to your PDF: ")
    print("Loading and indexing PDF...")
    docs = load_pdf(pdf_path)
    print("Building search index (first run may take a minute to download model)...")
    chain = build_qa_chain(docs)
    print("Ready! Ask questions about your PDF (type 'quit' to exit)\n")
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        answer = chain.invoke(question)
        print(f"AI: {answer}\n")