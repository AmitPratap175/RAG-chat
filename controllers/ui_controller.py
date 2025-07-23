import streamlit as st
import shutil
import os
from controllers.vector_db_manager import VectorDBManager
from controllers.document_processor import DocumentProcessor
from controllers.qa_graph_handler import CustomerSupportBot
from config import Config

class UIController:
    """
    UIController manages the Streamlit user interface for interacting with the Multi-Document RAG Assistant.

    Responsibilities:
    -----------------
    - Initialize and maintain document vector database state in Streamlit session state.
    - Provide controls for creating or using an existing document database.
    - Allow users to input queries and display answers retrieved via the QA handler.
    - Manage the layout and sidebar interactions within the Streamlit app.

    Attributes:
    -----------
    db_manager : VectorDBManager
        Manages vector database creation and access.
    doc_processor : DocumentProcessor
        Processes documents from directories into chunks for ingestion.
    qa_handler : CustomerSupportBot
        Handles question-answering queries using the vector store retriever.

    Methods:
    --------
    _init_session_state()
        Initializes Streamlit session state flags related to the database.
    _create_new_database(directory_path)
        Creates a new vector database from documents under the specified directory.
    _clear_database()
        Clears the persistent vector database and resets UI state.
    _show_database_controls(directory_path)
        Renders database management buttons in the sidebar.
    run()
        Runs the main Streamlit app loop handling the UI rendering and user interactions.
    """

    def __init__(self):
        """
        Initializes the UIController by instantiating core components and session state.
        """
        self.db_manager = VectorDBManager()
        self.doc_processor = DocumentProcessor()
        self.qa_handler = CustomerSupportBot()
        self._init_session_state()

    def _init_session_state(self):
        """
        Ensures that the Streamlit session state variable tracking database creation status
        is initialized to False if not present to coordinate UI logic.
        """
        if 'db_created' not in st.session_state:
            st.session_state.db_created = False

    def _create_new_database(self, directory_path):
        """
        Deletes any existing persistent vector database and creates a new database from
        document chunks extracted from the specified directory.

        Displays success feedback when creation completes.

        Parameters:
        -----------
        directory_path : str
            Filesystem path to the directory containing source PDF documents.
        """
        if os.path.exists(Config.PERSIST_DIR):
            shutil.rmtree(Config.PERSIST_DIR)
        chunks = self.doc_processor.process_directory(directory_path)
        if chunks:
            self.db_manager.create_db(chunks)
            st.session_state.db_created = True
            st.success(f"Database created with {len(chunks)} document chunks!")

    def _clear_database(self):
        """
        Removes the persistent vector database directory and resets the session state flag.

        Displays success feedback on database clearance.
        """
        if os.path.exists(Config.PERSIST_DIR):
            shutil.rmtree(Config.PERSIST_DIR)
            st.session_state.db_created = False
            st.success("Database cleared successfully!")
    
    def _show_database_controls(self, directory_path):
        """
        Renders two buttons in the sidebar to allow the user to either create a new vector database
        or to use an already existing one.

        Parameters:
        -----------
        directory_path : str
            Path to the directory containing PDFs to use for database creation.
        """
        col1, col3 = st.columns(2)
        with col1:
            if st.button("Create New"):
                self._create_new_database(directory_path)
        with col3:
            if st.button("Existing"):
                st.session_state.db_created = True
                st.success("Using existing database.")

    def run(self):
        """
        Runs the main Streamlit UI rendering and event loop.

        - Sets page configuration and title.
        - Renders sidebar for database management.
        - Depending on database state, shows query input or prompts to create a database.
        - Handles query submission, retrieves answers using the QA handler, and displays responses.
        """
        st.set_page_config(page_title="DocuBrain AI", layout="wide", page_icon=None)
        st.title("Multi-RAG - Multi-Document RAG Assistant")

        with st.sidebar:
            st.header("Database Management")
            directory_path = st.text_input("PDF Directory Path:")
            if directory_path and os.path.isdir(directory_path):
                # Directory exists; show controls to create or use database
                self._show_database_controls(directory_path)
            elif directory_path:
                # User entered an invalid path
                st.error("Invalid directory path!")

        if st.session_state.db_created:
            st.subheader("Ask Questions")
            query = st.text_input("Enter your question:")
            if st.button("Get Answer"):
                retriever = self.db_manager.get_retriever()
                # Retrieve answer from QA handler using the retriever
                response = self.qa_handler.handle_query(query, retriever)
                final_response = response['result'] if isinstance(response, dict) and 'result' in response else response
                st.markdown(final_response)
        else:
            st.info("Please create a document database using the sidebar")
