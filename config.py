import os
from dotenv import load_dotenv

# Load environment variables from .env file to configure secrets or environment-specific variables
load_dotenv()

class Config:
    """
    Configuration class for application-wide constants and environment variables.

    Attributes:
    -----------
    DATA_DIR : str
        Absolute path to the directory containing source data files.
    PERSIST_DIR : str
        Absolute path to the persistent vector database directory for Chroma.
    CHUNK_SIZE : int
        Number of characters per document chunk for processing and embedding.
    CHUNK_OVERLAP : int
        Number of characters that overlap between consecutive chunks for contextual continuity.
    MODEL_NAME : str
        Name of the large language model used for generative tasks.
    EMBEDDING_MODEL : str
        Path or identifier for the embeddings model.
    GOOGLE_API_KEY : str or None
        Google API key loaded from environment, used to authenticate Google-based models or services.
    PROMPT_FILE : str
        Absolute path to the prompt template text file for QA or chat operations.

    Usage:
    ------
    This class provides a single-point source of truth for all critical configuration settings
    in this application. Any component requiring config access should use these attributes
    to maintain consistency and enable centralized management.

    Example:
    --------
        data_path = Config.DATA_DIR
        api_key = Config.GOOGLE_API_KEY
    """
    DATA_DIR = os.getcwd() + "/data"
    PERSIST_DIR = os.getcwd() + "/chroma_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MODEL_NAME = "gemini-2.0-flash"
    EMBEDDING_MODEL = "models/embedding-001"
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PROMPT_FILE = os.getcwd() + "/templates/qa_prompt.txt"
