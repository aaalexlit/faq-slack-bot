from pathlib import Path
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def ingest_docs():
    loader = GoogleDriveLoader(service_account_key=Path.cwd() / "keys" / "service_account_key.json",
                               document_ids=["12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0"])

    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("slack_bot/faiss_index")

if __name__ == "__main__":
    ingest_docs()

