import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    DATA_DIR = os.getcwd() + "/data"
    PERSIST_DIR = os.getcwd() + "/chroma_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MODEL_NAME = "gemini-2.0-flash"
    EMBEDDING_MODEL = "models/embedding-001"
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    PROMPT_FILE = os.getcwd() + "/templates/qa_prompt.txt"
