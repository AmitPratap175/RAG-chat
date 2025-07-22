import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config

load_dotenv()

class VectorDBManager:
    def __init__(self):
        self.data_dir = Config.DATA_DIR 
        self.vectorstore_path = Config.PERSIST_DIR
        if not os.path.exists(self.vectorstore_path) or not os.listdir(self.vectorstore_path):
            print("Vectorstore not found or empty. Creating it first.")
            document_tuples = self.load_documents_from_directory()
            print(f"Loaded {len(document_tuples)} files from data directory.")

            self.create_vectorstore_incrementally(document_tuples)

    def load_documents_from_directory(self):
        documents = []
        for filename in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, filename)
            try:
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                elif filename.endswith(".txt"):
                    loader = TextLoader(filepath)
                else:
                    continue  # skip unsupported file types
                docs = loader.load()
                documents.append((filename, docs))  # Keep track of filename + docs
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
        return documents

    def create_vectorstore_incrementally(self, document_tuples):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Initialize or load a Chroma vectorstore, empty or existing at persist_dir
        if os.path.exists(self.vectorstore_path) and os.listdir(self.vectorstore_path):
            vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=embeddings
            )
        else:
            vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=embeddings
            )

        for filename, docs in document_tuples:
            print(f"Processing {filename} ({len(docs)} document(s))...")
            chunks = splitter.split_documents(docs)
            print(f"  Split into {len(chunks)} chunks.")
            vectorstore.add_documents(chunks)  # Incrementally add
            # Persistence is automatic; no need to call persist()
            print(f"  Added chunks from {filename} to the vectorstore.")

        print("Finished adding all documents incrementally.")
        return vectorstore

    def get_retriever(self):
        if not os.path.exists(self.vectorstore_path) or not os.listdir(self.vectorstore_path):
            print("Vectorstore not found or empty.")
            return
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        retriever = Chroma(
            persist_directory=self.vectorstore_path,
            embedding_function=embeddings
        ).as_retriever()
        return retriever

    