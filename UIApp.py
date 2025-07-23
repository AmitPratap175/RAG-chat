import shutil
import os
from controllers.vector_db_manager import VectorDBManager
from controllers.document_processor import DocumentProcessor
from controllers.qa_graph_handler import CustomerSupportBot
from config import Config
from cust_logger import logger
from fastapi import WebSocket
import json

class UIController:
    """
    UIController orchestrates interaction between the vector database manager, document processing,
    and the QA graph handler for managing and responding to user queries.

    Responsibilities:
    -----------------
    - Initialize and maintain state of the vector database
    - Process documents for ingestion if needed
    - Clear and rebuild vector database on demand
    - Handle real-time user queries through WebSocket communication

    Attributes:
    -----------
    db_manager : VectorDBManager
        Manages creation, loading, and access to the vector database.
    doc_processor : DocumentProcessor
        Responsible for processing and chunking documents from file system.
    qa_handler : CustomerSupportBot
        Handles query processing logic using a QA graph for customer support.
    db_created : bool
        Tracks whether the vector database has been created in current session.

    Methods:
    --------
    _init_state()
        Checks if the vector database exists and sets initialization state.
    _create_new_database(directory_path)
        Clears existing database and builds a new one from documents in the given directory.
    _clear_database()
        Deletes the persistent vector database and updates internal state.
    invoke_our_graph(websocket, data, user_uuid)
        Asynchronously handles incoming queries, processes them using QA handler and sends back streaming responses.
    """

    def __init__(self):
        """
        Initializes UIController by creating instances of core components and
        initializing database state.
        """
        self.db_manager = VectorDBManager()
        self.doc_processor = DocumentProcessor()
        self.qa_handler = CustomerSupportBot()
        self._init_state()

    def _init_state(self):
        """
        Initialize internal state by checking if persistent vector database path exists.

        Sets:
        -----
        self.db_created : bool
            True if vectorstore directory exists, False otherwise.

        On false, triggers creation of a new vector database from PDF directory configured.
        """
        if not os.path.exists(Config.PERSIST_DIR):
            self.db_created = False
            self._create_new_database(Config.PDF_DIR)
        else:
            self.db_created = True

    def _create_new_database(self, directory_path):
        """
        Clears any existing vector database and creates a new one from documents under given directory.

        Parameters:
        -----------
        directory_path : str
            Absolute or relative path to directory containing documents to ingest.

        Process:
        --------
        - Deletes existing persistent storage directory if present.
        - Uses DocumentProcessor to process and chunk documents.
        - Creates new vectorstore with the processed chunks via VectorDBManager.
        - Updates internal state flag.
        - Logs success message with number of chunks ingested.
        """
        if os.path.exists(Config.PERSIST_DIR):
            shutil.rmtree(Config.PERSIST_DIR)
            self.db_created = False
        chunks = self.doc_processor.process_directory(directory_path)
        if chunks:
            self.db_manager.create_db(chunks)
            self.db_created = True
            logger.info(f"Database created with {len(chunks)} document chunks!")

    def _clear_database(self):
        """
        Clears the persistent vector database by removing the storage directory
        and updates internal state accordingly.

        Logs the successful clearance of the database.
        """
        if os.path.exists(Config.PERSIST_DIR):
            shutil.rmtree(Config.PERSIST_DIR)
            self.db_created = False
            logger.info("Database cleared successfully!")

    async def invoke_our_graph(self, websocket: WebSocket, data: str, user_uuid: str):
        """
        Handles incoming user query by retrieving response from the QA graph
        and sending the response back asynchronously over WebSocket.

        Parameters:
        -----------
        websocket : WebSocket
            The active WebSocket connection to communicate with the client.
        data : str
            User's input/query string to process.
        user_uuid : str
            Unique identifier for the user's conversation session.

        Process:
        --------
        - Obtains retriever interface from vector database manager.
        - Passes user query and retriever to QA handler to get answer.
        - Extracts final result text safely from the handler's response.
        - Sends back the answer message as JSON to client over WebSocket.

        Exception Handling:
        -------------------
        Catches and prints any exception during the query handling process to avoid crash.
        """
        # Example of possible extra configuration or initial input (currently commented out)
        # initial_input = {"messages": data}
        # thread_config = {"configurable": {"thread_id": user_uuid}}
        # final_text = ""
        try:
            retriever = self.db_manager.get_retriever()
            response = self.qa_handler.handle_query(data, retriever)
            final_response = response['result'] if isinstance(response, dict) and 'result' in response else response

            message = json.dumps({"on_chat_model_stream": final_response})
            await websocket.send_text(message)
        except Exception as e:
            print(f'Exception: {e}')
