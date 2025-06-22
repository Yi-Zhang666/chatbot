from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import os

# ✅ Load all supported documents from a folder
def load_data(folder_path: str):
    docs = []
    print(f"📂 Scanning folder: {folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            print(f"📄 Processing file: {filename}")
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            print(f"📄 Processing file: {filename}")
            loader = TextLoader(file_path, encoding="utf-8")
        elif filename.endswith(".docx"):
            print(f"📄 Processing file: {filename}")
            loader = Docx2txtLoader(file_path)
        else:
            print(f"⛔ Skipping unsupported file: {filename}")
            continue
        docs.extend(loader.load())
    print(f"✅ Loaded {len(docs)} document chunks from {folder_path}")
    return docs

# ✅ Preprocess documents into smaller chunks
def preprocess_data(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"📎 Preprocessed into {len(chunks)} chunks.")
    print("✅ First chunk preview:")
    print(chunks[0].page_content[:500])
    return chunks


