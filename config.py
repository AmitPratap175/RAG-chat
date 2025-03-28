import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    PERSIST_DIR = "/home/dspratap/Documents/RAG-chat/database/chroma_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MODEL_NAME = "gemini-1.5-flash"
    EMBEDDING_MODEL = "models/embedding-001"
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    PROMPT_FILE = "./templates/qa_prompt.txt"
