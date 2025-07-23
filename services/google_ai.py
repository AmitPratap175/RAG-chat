from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from config import Config

class GoogleAI:
    """
    Wrapper class to provide configured instances of Google Generative AI clients.

    Responsibilities:
    -----------------
    - Provide a Language Model (LLM) client instance configured with model name, temperature, and API key.
    - Provide an embeddings client instance configured with embedding model and API key.

    Usage:
    ------
    Instantiate GoogleAI and call `get_llm()` to retrieve the chat LLM client,
    or `get_embeddings()` to retrieve the embeddings client.
    """

    def get_llm(self):
        """
        Instantiate and return a ChatGoogleGenerativeAI client.

        Configuration:
        --------------
        - model: Uses the model name from Config.MODEL_NAME
        - temperature: Fixed at 0.5 for balanced response creativity
        - google_api_key: Pulled from Config environment variable

        Returns:
        --------
        ChatGoogleGenerativeAI
            Configured language model client for chat/completions.
        """
        return ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=0.5,
            google_api_key=Config.GOOGLE_API_KEY
        )

    def get_embeddings(self):
        """
        Instantiate and return a GoogleGenerativeAIEmbeddings client.

        Configuration:
        --------------
        - model: Uses embedding model identifier from Config.EMBEDDING_MODEL
        - google_api_key: Pulled from Config environment variable

        Returns:
        --------
        GoogleGenerativeAIEmbeddings
            Configured embeddings client for vectorizing text.
        """
        return GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            google_api_key=Config.GOOGLE_API_KEY
        )
