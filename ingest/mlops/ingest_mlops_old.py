import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pinecone  # type: ignore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain_community.document_loaders import GoogleDriveLoader, GitLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
embedding_dimension = len(embeddings.embed_query("test"))
print(f'embedding dimension = {embedding_dimension}')


def ingest_google_doc(index_name: str,
                      document_ids: list[str],
                      ):
    print('Loading google doc...')
    temp_creds = tempfile.NamedTemporaryFile()
    creds_dict = os.getenv('GOOGLE_SERVICE_ACC_KEY')
    with open(temp_creds.name, 'w') as f_out:
        json.dump(creds_dict, f_out)
    loader = GoogleDriveLoader(service_account_key=Path(temp_creds.name),
                               document_ids=document_ids)

    raw_docs = loader.load()
    temp_creds.close()
    print('Splitting docs for indexing...')
    text_splitter = get_text_splitter()
    docs = text_splitter.split_documents(raw_docs)

    index_docs(docs, index_name)


def index_docs(docs, index_name):
    print('Filling the index up...')
    Pinecone.from_documents(docs, embeddings, index_name=index_name)
    time.sleep(10)
    print_index_status(index_name)


def create_pinecone_index(index_name: str):
    if index_name in pinecone.list_indexes():
        print(f"Index {index_name} exists. Deleting...")
        pinecone.delete_index(index_name)

    if index_name not in pinecone.list_indexes():
        print(f"Creating index {index_name}...")
        pinecone.create_index(
            name=index_name,
            dimension=embedding_dimension
        )

    print_index_status(index_name)


def print_index_status(index_name):
    index = pinecone.GRPCIndex(index_name)
    index_stats = index.describe_index_stats()
    print(f"index stats: {index_stats}")


def ingest_git_repo(repo_url: str, index_name: str):
    local_dir_path = f"./git/{repo_url[repo_url.rindex('/') + 1:]}"
    if Path(local_dir_path).exists():
        remove_local_dir(local_dir_path)
    loader = GitLoader(
        clone_url=repo_url,
        repo_path=local_dir_path,
    )
    print('Loading and Splitting git repo for indexing...')
    text_splitter = get_text_splitter()
    docs = loader.load_and_split(text_splitter)
    index_docs(docs, index_name)
    remove_local_dir(local_dir_path)


def remove_local_dir(local_dir_path):
    print(f'Removing local files in {local_dir_path}')
    shutil.rmtree(local_dir_path)


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )


def create_and_fill_the_index(index_name: str,
                              google_doc_ids: list[str],
                              repo_url: str,
                              overwrite: bool):
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
    )
    if overwrite:
        create_pinecone_index(index_name=index_name)
    ingest_google_doc(index_name,
                      google_doc_ids)
    ingest_git_repo(repo_url, index_name)


if __name__ == "__main__":
    index_name = 'mlops-faq-bot'
    google_doc_id = ["12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0"]
    repo_url = 'https://github.com/DataTalksClub/mlops-zoomcamp'
    overwrite = True
    create_and_fill_the_index(index_name=index_name,
                              google_doc_ids=google_doc_id,
                              repo_url=repo_url,
                              overwrite=overwrite)
