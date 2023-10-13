#!/usr/bin/env python3
import os
import glob
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm

import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    #JSONLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

from constants import CHROMA_SETTINGS
import chromadb
from chromadb.api.segment import API

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50

# Create a function to extract and filter JSON data
def extract_and_filter_json(json_file):
    with open(json_file, "r") as json_file:
        data = json.load(json_file)

    filtered_data = []

    for item in data:
        if all(key in item for key in ["type", "text", "ts", "user"]):
            user_profile = item.get("user_profile", {})
            user_name = user_profile.get("name", "")
            ts_to_date = datetime.utcfromtimestamp(float(item['ts']))
            date_time = ts_to_date.strftime("%m/%d/%Y, %H:%M")

            filtered_data.append({
                "type": item["type"],
                "text": item["text"],
                "ts": date_time,
                "user": item["user"],
                "user_name": user_name
            })

    return filtered_data
def json_slack_to_pdf():
    # Get the current working directory
    current_directory = os.getcwd()

    # JSON directory path (modify as needed)
    json_directory = os.path.join(current_directory, "source_documents")

    # List of JSON files in the directory
    json_files = [os.path.join(json_directory, file) for file in os.listdir(json_directory) if file.endswith(".json")]

    # Create a PDF document
    pdf_filename = os.path.join(json_directory, "filtered_data.pdf")
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)

    # Create a list of paragraphs with the filtered data
    story = []

    # Define a style for the paragraphs
    styles = getSampleStyleSheet()
    style = styles["Normal"]

    # Process each JSON file in the directory
    for json_file in json_files:
        filtered_data = extract_and_filter_json(json_file)
    
        # Add the filtered data to the PDF
        for item in filtered_data:
            #text = f"Type: {item['type']}\nText: {item['text']}\nTimestamp: {item['ts']}\nUser: {item['user']}\nUser Name: {item['user_name']}\n\n"
            text = f"Text: {item['text']}\nTimestamp: {item['ts']}\nUser Name: {item['user_name']}\n\n"
            p = Paragraph(text, style)
            story.append(p)

    # Build the PDF document
    doc.build(story)
    # Delete the old JSON files
    #for json_file in json_files:
        #os.remove(json_file)

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    #".json": (JSONLoader, {"jq_schema":"text"})
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True)
        )
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.upper()}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)
    print(f"Split into {len(documents)} chunks of text (max. {chunk_size} tokens each)")
    return documents

def batch_chromadb_insertions(chroma_client: API, documents: List[Document]) -> List[Document]:
    """
    Split the total documents to be inserted into batches of documents that the local chroma client can process
    """
    # Get max batch size.
    max_batch_size = chroma_client.max_batch_size
    for i in range(0, len(documents), max_batch_size):
        yield documents[i:i + max_batch_size]


def does_vectorstore_exist(persist_directory: str, embeddings: HuggingFaceEmbeddings) -> bool:
    """
    Checks if vectorstore exists
    """
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    if not db.get()['documents']:
        return False
    return True

def main():
    json_slack_to_pdf()
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    # Chroma client
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)

    if does_vectorstore_exist(persist_directory, embeddings):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
        collection = db.get()
        documents = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        for batched_chromadb_insertion in batch_chromadb_insertions(chroma_client, documents):
            db.add_documents(batched_chromadb_insertion)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        documents = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        # Create the db with the first batch of documents to insert
        batched_chromadb_insertions = batch_chromadb_insertions(chroma_client, documents)
        first_insertion = next(batched_chromadb_insertions)
        db = Chroma.from_documents(first_insertion, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS, client=chroma_client)
        # Add the rest of batches of documents
        for batched_chromadb_insertion in batched_chromadb_insertions:
            db.add_documents(batched_chromadb_insertion)

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()
