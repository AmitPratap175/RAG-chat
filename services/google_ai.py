from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from config import Config

class GoogleAI:
    def get_llm(self):
        return ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=0.5,
            google_api_key=Config.GOOGLE_API_KEY
        )

    def get_embeddings(self):
        return GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            google_api_key=Config.GOOGLE_API_KEY
        )
