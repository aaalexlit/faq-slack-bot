from pathlib import Path
import os
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone  # type: ignore
from prefect import flow, task

@task(log_prints=True)
def ingest_google_doc(index_name: str,
                      document_ids: list[str],
                      ):
    print('Loading google doc...')
    loader = GoogleDriveLoader(service_account_key=Path.cwd() / "keys" / "service_account_key.json",
                               document_ids=document_ids)

    raw_docs = loader.load()
    print('Splitting docs for indexing...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(raw_docs)

    print('Filling the index up...')
    embeddings = HuggingFaceEmbeddings()
    Pinecone.from_documents(docs, embeddings, index_name=index_name)
    print_index_status(index_name)

@task(log_prints=True)
def create_pinecone_index(index_name: str):
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
    )

    if index_name in pinecone.list_indexes():
        print(f"Index {index_name} exists. Deleting...")
        pinecone.delete_index(index_name)

    if index_name not in pinecone.list_indexes():
        print(f"Creating index {index_name}...")
        pinecone.create_index(
            name=index_name,
            dimension=768
        )

    print_index_status(index_name)


def print_index_status(index_name):
    index = pinecone.GRPCIndex(index_name)
    index_stats = index.describe_index_stats()
    print(f"index stats: {index_stats}")

@flow()
def create_and_fill_the_index(index_name: str,
                              google_doc_ids: list[str]):
    create_pinecone_index(index_name=index_name)
    ingest_google_doc(index_name,
                      google_doc_ids)


if __name__ == "__main__":
    index_name = 'mlops-faq-bot'
    google_doc_id = ["12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0"]
    create_and_fill_the_index(index_name=index_name,
                              google_doc_ids=google_doc_id)
