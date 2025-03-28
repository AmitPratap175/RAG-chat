import streamlit as st
import shutil
import os
from controllers.vector_db_manager import VectorDBManager
from controllers.document_processor import DocumentProcessor
from controllers.qa_graph_handler import CustomerSupportBot
from config import Config

class UIController:
    def __init__(self):
        self.db_manager = VectorDBManager()
        self.doc_processor = DocumentProcessor()
        self.qa_handler = CustomerSupportBot()
        self._init_session_state()

    def _init_session_state(self):
        if 'db_created' not in st.session_state:
            st.session_state.db_created = False

    def _create_new_database(self, directory_path):
        if os.path.exists(Config.PERSIST_DIR):
            shutil.rmtree(Config.PERSIST_DIR)
        chunks = self.doc_processor.process_directory(directory_path)
        if chunks:
            self.db_manager.create_db(chunks)
            st.session_state.db_created = True
            st.success(f"Database created with {len(chunks)} document chunks!")

    def _clear_database(self):
        if os.path.exists(Config.PERSIST_DIR):
            shutil.rmtree(Config.PERSIST_DIR)
            st.session_state.db_created = False
            st.success("Database cleared successfully!")
    
    def _show_database_controls(self, directory_path):
        col1, col3 = st.columns(2)
        with col1:
            if st.button("Create New"):
                self._create_new_database(directory_path)
        with col3:
            if st.button("Existing"):
                st.session_state.db_created = True
                st.success("Using existing database.")

    def run(self):
        st.set_page_config(page_title="DocuBrain AI", layout="wide", page_icon="🧠")
        st.title("Multi-RAG - Multi-Document RAG Assistant")

        with st.sidebar:
            st.header("⚙️ Database Management")
            directory_path = st.text_input("📂 PDF Directory Path:")
            if directory_path and os.path.isdir(directory_path):
                if os.path.isdir(directory_path):
                    self._show_database_controls(directory_path)
                else:
                    st.error("Invalid directory path!")

        if st.session_state.db_created:
            st.subheader("🔍 Ask Questions")
            query = st.text_input("Enter your question:")
            if st.button("🚀 Get Answer"):
                retriever = self.db_manager.get_retriever()
                response = self.qa_handler.handle_query(query, retriever)
                print(f"\n\n\nResponse: {response}\n\n\n")
                final_response = response['result'] if isinstance(response, dict) and 'result' in response else response
                st.markdown(final_response)
        else:
            st.info("💡 Please create a document database using the sidebar")
