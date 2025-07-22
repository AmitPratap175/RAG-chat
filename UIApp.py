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
    def __init__(self):
        self.db_manager = VectorDBManager()
        self.doc_processor = DocumentProcessor()
        self.qa_handler = CustomerSupportBot()
        self._init_state()

    def _init_state(self):
        if not os.path.exists(Config.PERSIST_DIR):
            self.db_created = False
            self._create_new_database(Config.PDF_DIR)
        else:
            self.db_created = True

    def _create_new_database(self, directory_path):
        if os.path.exists(Config.PERSIST_DIR):
            shutil.rmtree(Config.PERSIST_DIR)
            self.db_created = False
        chunks = self.doc_processor.process_directory(directory_path)
        if chunks:
            self.db_manager.create_db(chunks)
            self.db_created = True
            logger.info(f"Database created with {len(chunks)} document chunks!")

    def _clear_database(self):
        if os.path.exists(Config.PERSIST_DIR):
            shutil.rmtree(Config.PERSIST_DIR)
            self.db_created = False
            logger.info("Database cleared successfully!")

    async def invoke_our_graph(self, websocket: WebSocket, data: str, user_uuid: str):
        # initial_input = {"messages": data}
        # thread_config = {"configurable": {"thread_id": user_uuid}}
        # final_text = ""
        try:
            # self.db_manager.create_db()
            retriever = self.db_manager.get_retriever()
            response = self.qa_handler.handle_query(data, retriever)
            final_response = response['result'] if isinstance(response, dict) and 'result' in response else response
            
            message = json.dumps({"on_chat_model_stream": final_response})
            await websocket.send_text(message)
        except Exception as e:
            print(f'Exception: {e}')

