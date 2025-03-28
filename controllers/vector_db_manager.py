import os
from services.google_ai import GoogleAI
from langchain_chroma import Chroma
from config import Config

class VectorDBManager:
    def __init__(self):
        self.embeddings = GoogleAI().get_embeddings()

    def create_db(self, chunks):
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=Config.PERSIST_DIR
        )

    def get_retriever(self):
        if os.path.exists(Config.PERSIST_DIR):
            return Chroma(
                persist_directory=Config.PERSIST_DIR,
                embedding_function=self.embeddings
            ).as_retriever(search_kwargs={"k": 3})
        return None
