import json
import os
import tempfile
from datetime import datetime, timedelta

from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.storage import UpstashRedisByteStore
from llama_index import Document, StorageContext, ServiceContext, VectorStoreIndex
from llama_index.node_parser import NodeParser, SentenceSplitter
from llama_index.readers import TrafilaturaWebReader, GithubRepositoryReader
from llama_index.vector_stores import MilvusVectorStore
from prefect.blocks.system import Secret
from prefect_gcp import GcpCredentials
from upstash_redis import Redis

from ingest.readers.custom_faq_gdoc_reader import FAQGoogleDocsReader
from ingest.readers.slack_reader import SlackReader

BOT_USER_ID = 'U05DM3PEJA2'
AU_TOMATOR_USER_ID = 'U01S08W6Z9T'

os.environ["TOKENIZERS_PARALLELISM"] = "false"

embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')

embedding_dimension = len(embeddings.embed_query("test"))
print(f'embedding dimension = {embedding_dimension}')


def index_spreadsheet(url: str, title: str, collection_name: str):
    documents = TrafilaturaWebReader().load_data([url])
    for doc in documents:
        doc.metadata['title'] = title
        doc.metadata['source'] = url
    add_route_to_docs(documents, 'faq')
    add_to_index(documents, collection_name=collection_name)


def add_route_to_docs(docs: [Document], route_name: str):
    route_key_name = 'route'
    for doc in docs:
        doc.metadata[route_key_name] = route_name
        doc.excluded_embed_metadata_keys.append(route_key_name)
        doc.excluded_llm_metadata_keys.append(route_key_name)


def add_to_index(documents: [Document],
                 collection_name: str,
                 overwrite: bool = False,
                 node_parser: NodeParser = None):
    if not node_parser:
        node_parser = SentenceSplitter.from_defaults(chunk_size=512, chunk_overlap=50)
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
    service_context = ServiceContext.from_defaults(embed_model=load_embeddings(),
                                                   node_parser=node_parser,
                                                   llm=None)
    VectorStoreIndex.from_documents(documents,
                                    storage_context=storage_context,
                                    service_context=service_context,
                                    show_progress=True)


def index_github_repo(owner: str,
                      repo: str,
                      branch: str,
                      collection_name: str,
                      ignore_file_extensions: [str] = None,
                      ignore_directories: [str] = None,
                      ):
    if ignore_file_extensions is None:
        ignore_file_extensions = ['.jpg', '.png', '.gitignore', '.csv']
    if ignore_directories is None:
        ignore_directories = ['.github', '.gitignore', '2021', '2022', 'images']
    documents = GithubRepositoryReader(
        owner=owner,
        repo=repo,
        github_token=Secret.load('github-token').get(),
        ignore_file_extensions=ignore_file_extensions,
        ignore_directories=ignore_directories,
    ).load_data(branch=branch)
    for doc in documents:
        doc.metadata['branch'] = branch
        doc.metadata['owner'] = owner
        doc.metadata['repo'] = repo
    add_route_to_docs(documents, 'github')
    add_to_index(documents, collection_name=collection_name)


def index_slack_history(channel_ids: [str], collection_name: str):
    earliest_date = datetime.now() - timedelta(days=90)
    slack_reader = SlackReader(earliest_date=earliest_date,
                               bot_user_id=BOT_USER_ID,
                               not_ignore_users=[AU_TOMATOR_USER_ID],
                               slack_token=Secret.load('slack-bot-token').get())
    documents = slack_reader.load_data(channel_ids=channel_ids)
    add_route_to_docs(documents, 'slack')
    add_to_index(documents,
                 collection_name=collection_name,
                 overwrite=False,
                 )


def index_faq(document_ids: [str], collection_name: str, question_heading_style_num: int):
    temp_creds = tempfile.NamedTemporaryFile()
    creds_dict = GcpCredentials.load("google-drive-creds").service_account_info.get_secret_value()
    with open(temp_creds.name, 'w') as f_out:
        json.dump(creds_dict, f_out)
    gdocs_reader = FAQGoogleDocsReader(service_account_json_path=temp_creds.name,
                                       question_heading_style_num=question_heading_style_num)
    documents = gdocs_reader.load_data(document_ids=document_ids)
    temp_creds.close()
    add_route_to_docs(documents, 'faq')
    add_to_index(documents,
                 collection_name=collection_name,
                 overwrite=True,
                 )


def load_embeddings() -> CacheBackedEmbeddings:
    redis_client = Redis(url=Secret.load('upstash-redis-rest-url').get(),
                         token=Secret.load('upstash-redis-rest-token').get())
    embeddings_cache = UpstashRedisByteStore(client=redis_client,
                                             ttl=None,
                                             namespace=os.getenv('EMBEDDING_CACHE_NAMESPACE'))

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        embeddings_cache,
        namespace=embeddings.model_name + "/",
    )
    return cached_embedder
