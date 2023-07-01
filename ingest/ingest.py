import json
import time
from pathlib import Path
import shutil
import os
import tempfile

from langchain.document_loaders import GoogleDriveLoader, GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone  # type: ignore

from prefect import flow, task
from prefect_gcp import GcpCredentials
from prefect.blocks.system import Secret

os.environ["TOKENIZERS_PARALLELISM"] = "false"
embeddings = HuggingFaceEmbeddings()


@task(name="Index FAQ Google Document", log_prints=True)
def ingest_google_doc(index_name: str,
                      document_ids: list[str],
                      ):
    print('Loading google doc...')
    temp_creds = tempfile.NamedTemporaryFile()
    creds_dict = GcpCredentials.load("google-drive-creds").service_account_info.get_secret_value()
    with open(temp_creds.name, 'w') as f_out:
        json.dump(creds_dict, f_out)
    loader = GoogleDriveLoader(service_account_key=Path(temp_creds.name),
                               document_ids=document_ids)
    # loader = GoogleDriveLoader(service_account_key=Path.cwd() / "keys" / "service_account_key.json",
    #                            document_ids=document_ids)

    raw_docs = loader.load()
    temp_creds.close()
    print('Splitting docs for indexing...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(raw_docs)

    index_docs(docs, index_name)


def index_docs(docs, index_name):
    print('Filling the index up...')
    Pinecone.from_documents(docs, embeddings, index_name=index_name)
    time.sleep(10)
    print_index_status(index_name)


@task(name="Delete and Create Pinecone index", log_prints=True)
def create_pinecone_index(index_name: str):
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


@task(name="Index git repo", log_prints=True)
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


@flow(name="Update the index")
def create_and_fill_the_index(index_name: str,
                              google_doc_ids: list[str],
                              repo_url: str,
                              overwrite: bool):
    pinecone.init(
        api_key=Secret.load('pinecone-api-key').get(),
        environment=Secret.load('pinecone-env').get()
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
