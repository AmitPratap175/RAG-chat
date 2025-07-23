import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.pdf_loader import PDFLoader
from config import Config

class DocumentProcessor:
    """
    DocumentProcessor is responsible for processing PDF documents in a directory,
    splitting them into chunks suitable for downstream embedding or analysis.

    Responsibilities:
    -----------------
    - Configure a text splitter for document chunking.
    - Load and split all PDF documents in a given directory.
    - Aggregate all resulting document chunks for use in vectorization or search.

    Attributes:
    -----------
    splitter : RecursiveCharacterTextSplitter
        Configured text splitter for segmenting documents into manageable pieces.

    Methods:
    --------
    process_directory(directory_path)
        Processes all PDF files in the specified directory, returning a list of all chunks.
    """

    def __init__(self):
        """
        Initialize DocumentProcessor by setting up the text splitter
        using chunk size and overlap values from the configuration.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

    def process_directory(self, directory_path):
        """
        Process all PDF files found in the provided directory path.

        For each PDF, loads the document, splits it into text chunks,
        and adds these chunks to the output list.

        Parameters:
        -----------
        directory_path : str
            Filesystem path to the directory containing PDF files.

        Returns:
        --------
        list
            A list of document chunks obtained from all PDFs in the directory.
        """
        all_docs = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(directory_path, filename)
                loader = PDFLoader(file_path)
                docs = loader.load_and_split(self.splitter)
                all_docs.extend(docs)
        return all_docs
