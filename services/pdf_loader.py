from langchain_community.document_loaders import PyPDFLoader

class PDFLoader:
    """
    PDFLoader is a wrapper for loading and splitting PDF documents using LangChain's PyPDFLoader.

    Attributes:
    -----------
    file_path : str
        Path to the PDF file to be loaded.

    Methods:
    --------
    load_and_split(splitter)
        Loads the PDF and splits it into chunks using the provided text splitter.
    """

    def __init__(self, file_path: str):
        """
        Initializes the PDFLoader with the target PDF file path.

        Parameters:
        -----------
        file_path : str
            The filesystem path to the PDF document to be processed.
        """
        self.file_path = file_path

    def load_and_split(self, splitter):
        """
        Loads the PDF document and splits its content into smaller chunks using a text splitter.

        Parameters:
        -----------
        splitter : TextSplitter
            An instance of a text splitter class (e.g., RecursiveCharacterTextSplitter)
            to segment the raw document content into manageable pieces.

        Returns:
        --------
        List[Document]
            A list of document chunks generated by splitting the loaded PDF content.
        """
        loader = PyPDFLoader(self.file_path)
        return loader.load_and_split(splitter)
