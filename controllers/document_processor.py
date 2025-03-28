import os
import datetime
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.pdf_loader import PDFLoader
from config import Config

class DocumentProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

    def process_directory(self, directory_path):
        """Process all PDFs in a directory"""
        all_docs = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(directory_path, filename)
                loader = PDFLoader(file_path)
                docs = loader.load_and_split(self.splitter)
                all_docs.extend(docs)
        return all_docs
