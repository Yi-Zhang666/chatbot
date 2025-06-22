from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

def preprocess_data(documents: List[Document]) -> List[Document]:
    # Define chunk size and overlap (you can adjust these values)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Split documents into smaller chunks
    split_docs = text_splitter.split_documents(documents)

    print(f"📎 Preprocessed into {len(split_docs)} chunks.")
    return split_docs

# Optional: Run as a standalone script for testing
if __name__ == "__main__":
    from data_loader import load_data
    docs = load_data("data")
    chunks = preprocess_data(docs)

    print(f"✅ First chunk preview:\n{chunks[0].page_content[:300]}")
