import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from langchain.document_loaders import GoogleDriveLoader
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import StorageContext, VectorStoreIndex, ServiceContext, TrafilaturaWebReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.readers import Document
from llama_index.vector_stores import MilvusVectorStore
from prefect import flow, task
from prefect.blocks.system import Secret
from prefect_gcp import GcpCredentials

from slack_reader import SlackReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

embeddings = HuggingFaceEmbeddings()

embedding_dimension = len(embeddings.embed_query("test"))
logger.info(f'embedding dimension = {embedding_dimension}')

BOT_USER_ID = 'U05DM3PEJA2'
AU_TOMATOR_USER_ID = 'U01S08W6Z9T'
ML_CHANNEL_ID = 'C0288NJ5XSA'
FAQ_COLLECTION_NAME = 'mlzoomcamp_faq_git'
SLACK_COLLECTION_NAME = 'mlzoomcamp_slack'


@task(name="Index FAQ Google Document")
def index_google_doc():
    document_ids = ["1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8"]
    logger.info('Loading google doc...')
    temp_creds = tempfile.NamedTemporaryFile()
    creds_dict = GcpCredentials.load("google-drive-creds").service_account_info.get_secret_value()
    with open(temp_creds.name, 'w') as f_out:
        json.dump(creds_dict, f_out)
    loader = GoogleDriveLoader(service_account_key=Path(temp_creds.name),
                               document_ids=document_ids)
    raw_docs = loader.load()
    temp_creds.close()
    add_to_index([Document.from_langchain_format(doc) for doc in raw_docs],
                 collection_name=FAQ_COLLECTION_NAME)


@task(name="Index course schedule")
def index_course_schedule():
    url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSkEwMv5OKwCdPfW6LgqQvKk48dZjPcFDrjDstBqZfq38UPadh0Nws1b57qOVYwzAjSufKnVf7umGWH/pubhtml'
    documents = TrafilaturaWebReader().load_data([url])
    for doc in documents:
        doc.metadata['title'] = 'ML Zoomcamp 2023 syllabus and deadlines'
        doc.metadata['source'] = url
    add_to_index(documents,
                 collection_name=FAQ_COLLECTION_NAME,
                 overwrite=False
                 )


@task(name="Index slack messages")
def index_slack_messages():
    slack_reader = SlackReader(earliest_date=datetime(2022, 9, 1),
                               bot_user_id=BOT_USER_ID,
                               not_ignore_users=[AU_TOMATOR_USER_ID],
                               slack_token=Secret.load('slack-bot-token').get())

    documents = slack_reader.load_data(channel_ids=[ML_CHANNEL_ID])
    add_to_index(documents, collection_name=SLACK_COLLECTION_NAME)


def add_to_index(documents, collection_name, overwrite=True):
    node_parser = SimpleNodeParser.from_defaults(chunk_size=384, chunk_overlap=20)
    environment = os.getenv('EXECUTION_ENV', 'local')
    if environment == 'local':
        milvus_vector_store = MilvusVectorStore(collection_name=collection_name,
                                                dim=embedding_dimension,
                                                overwrite=overwrite)
    else:
        milvus_vector_store = MilvusVectorStore(collection_name=collection_name,
                                                uri=Secret.load('zilliz-cloud-uri').get(),
                                                token=Secret.load('zilliz-cloud-api-key').get(),
                                                dim=embedding_dimension,
                                                overwrite=overwrite)
    storage_context = StorageContext.from_defaults(vector_store=milvus_vector_store)
    service_context = ServiceContext.from_defaults(embed_model=embeddings, node_parser=node_parser, llm=None)
    VectorStoreIndex.from_documents(documents, storage_context=storage_context,
                                    service_context=service_context)


@flow(name="Update ML info Milvus index", log_prints=True)
def fill_ml_index():
    index_google_doc()
    index_course_schedule()
    index_slack_messages()


# if __name__ == '__main__':
#     fill_ml_index()
