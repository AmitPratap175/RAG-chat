import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config

# Load environment variables required for configuration and authentication.
# This should be called before accessing any secrets or environment-based paths to ensure
# they are available to the application.
load_dotenv()

class VectorDBManager:
    """
    VectorDBManager manages all aspects of document ingestion and retrieval for a vector database
    using the Chroma backend.

    Responsibilities:
    ---------------
    - Loads PDF and TXT documents from a specified directory
    - Splits and embeds documents using GoogleGenerativeAIEmbeddings
    - Builds or updates a persistent Chroma vectorstore
    - Provides a retriever interface for vector-based semantic search

    Parameters:
    -----------
    None (configuration is loaded from the 'Config' module)

    Methods:
    --------
    __init__()
        Initializes the instance, loads or creates the vectorstore as needed.
    load_documents_from_directory()
        Loads and parses all supported files from the data directory.
    create_vectorstore_incrementally(document_tuples)
        Chunks and embeds the loaded documents and populates the vectorstore.
    get_retriever()
        Returns a retriever object for performing semantic retrieval over the vectorstore.
    """

    def __init__(self):
        """
        Initializes the VectorDBManager.

        Reads configuration from the Config object:
            - self.data_dir: directory containing user documents to embed
            - self.vectorstore_path: path where Chroma vectorstore files are stored

        If the vectorstore is missing or empty, triggers initial ingestion by:
            - Loading all supported documents from the data directory
            - Incrementally building the vectorstore

        Best Practice:
        - Ensures that no vector retrieval happens without at least
          one initialization or ingestion cycle.
        """
        self.data_dir = Config.DATA_DIR
        self.vectorstore_path = Config.PERSIST_DIR

        # Check if the persistent vectorstore exists and is non-empty
        if not os.path.exists(self.vectorstore_path) or not os.listdir(self.vectorstore_path):
            print("Vectorstore not found or empty. Creating it first.")
            document_tuples = self.load_documents_from_directory()
            print(f"Loaded {len(document_tuples)} files from data directory.")
            self.create_vectorstore_incrementally(document_tuples)

    def load_documents_from_directory(self):
        """
        Loads and parses documents from the configured data directory.

        Supported formats:
            - PDF (handled by PyPDFLoader)
            - TXT (handled by TextLoader)

        Skips files with unsupported extensions.

        Returns:
            documents (list of tuples): List of (filename, [document_objects])
                where each document object is parsed by its respective loader.

        Error Handling:
            - If a file fails to load, logs the error and continues processing the rest.
        """
        documents = []
        for filename in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, filename)
            try:
                # Select and instantiate the loader based on the file extension
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                elif filename.endswith(".txt"):
                    loader = TextLoader(filepath)
                else:
                    continue  # Skip unsupported file types
                docs = loader.load()
                documents.append((filename, docs))  # Store for downstream vectorization
            except Exception as e:
                # Log error for traceability and diagnostics; do not break pipeline
                print(f"Failed to load {filename}: {e}")
        return documents

    def create_vectorstore_incrementally(self, document_tuples):
        """
        Chunks, embeds, and adds given documents to a (possibly existing) Chroma vectorstore.

        Parameters:
        -----------
        document_tuples : list
            List of (filename, [document_objects]) from the document loader.

        Steps:
        ------
        - Instantiate a RecursiveCharacterTextSplitter for chunking each document.
        - Use GoogleGenerativeAIEmbeddings for vector representation.
        - Initialize or load an existing Chroma vectorstore at persist_directory.
        - For each file:
            * Split into overlapping chunks
            * Add each chunk to the Chroma store for later retrieval

        Returns:
            vectorstore (Chroma): The ready-to-use vectorstore instance.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Chroma vectorstore is initialized or loaded from path for incremental ingestion
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
            # Split into semantic chunks for better vector granularity
            chunks = splitter.split_documents(docs)
            print(f"  Split into {len(chunks)} chunks.")
            # Each chunk is added to the vectorstore; persistence is managed by Chroma
            vectorstore.add_documents(chunks)
            print(f"  Added chunks from {filename} to the vectorstore.")

        print("Finished adding all documents incrementally.")
        return vectorstore

    def get_retriever(self):
        """
        Returns a retriever object for semantic search over the vectorstore.

        Ensures the vectorstore directory is present and non-empty before proceeding.

        Returns:
            retriever (Chroma.as_retriever): A retriever instance for use in downstream chains.
            Returns None if the vectorstore does not exist or is empty.

        Usage Consideration:
            - Uses the same embedding function as used during document ingestion for consistency.
        """
        if not os.path.exists(self.vectorstore_path) or not os.listdir(self.vectorstore_path):
            print("Vectorstore not found or empty.")
            return
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        retriever = Chroma(
            persist_directory=self.vectorstore_path,
            embedding_function=embeddings
        ).as_retriever()
        return retriever
