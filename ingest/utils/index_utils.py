import json
import os
import re
import tempfile
from datetime import datetime, timedelta

from jupyter_notebook_parser import JupyterNotebookParser
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.storage import UpstashRedisByteStore
from llama_index.core import Settings
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.node_parser import NodeParser, SentenceSplitter, MarkdownNodeParser
from llama_index.core.schema import Document
from llama_index.core.storage import StorageContext
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from upstash_redis import Redis

from ingest.readers.slack_reader import SlackReader
from ingest.readers.youtube_reader import YoutubeReader

BOT_USER_ID = 'U05DM3PEJA2'
AU_TOMATOR_USER_ID = 'U01S08W6Z9T'

EXCLUDE_FILTER_TYPE = GithubRepositoryReader.FilterType.EXCLUDE

os.environ["TOKENIZERS_PARALLELISM"] = "false"

embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')

embedding_dimension = len(embeddings.embed_query("test"))
print(f'embedding dimension = {embedding_dimension}')


def load_embeddings() -> CacheBackedEmbeddings:
    redis_client = Redis(url=os.getenv('UPSTASH_REDIS_REST_URL'),
                         token=os.getenv('UPSTASH_REDIS_REST_TOKEN'))
    embeddings_cache = UpstashRedisByteStore(client=redis_client,
                                             ttl=None,
                                             namespace=os.getenv('EMBEDDING_CACHE_NAMESPACE'))

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        embeddings_cache,
        namespace=embeddings.model_name + "/",
    )
    return cached_embedder


Settings.embed_model = load_embeddings()
Settings.llm = None


def index_spreadsheet(url: str, title: str, collection_name: str):
    documents = TrafilaturaWebReader().load_data([url])
    for doc in documents:
        doc.metadata['title'] = title
        doc.metadata['source'] = url
    add_route_to_docs(documents, 'faq')
    add_to_index(documents, collection_name=collection_name)


def add_route_to_docs(docs: list[Document], route_name: str):
    route_key_name = 'route'
    for doc in docs:
        doc.metadata[route_key_name] = route_name
        doc.excluded_embed_metadata_keys.append(route_key_name)
        doc.excluded_llm_metadata_keys.append(route_key_name)


def add_to_index(documents: list[Document],
                 collection_name: str,
                 overwrite: bool = False,
                 node_parser: NodeParser = None):
    sentence_splitter = SentenceSplitter.from_defaults(chunk_size=512, chunk_overlap=50,
                                                       tokenizer=embeddings.client.tokenizer)
    environment = os.getenv('EXECUTION_ENV', 'local')
    if environment == 'local':
        milvus_vector_store = MilvusVectorStore(uri='http://localhost:19530',
                                                collection_name=collection_name,
                                                dim=embedding_dimension,
                                                overwrite=overwrite)
    elif environment == 'zilliz-cluster':
        milvus_vector_store = MilvusVectorStore(
            uri=os.getenv('ZILLIZ_PUBLIC_ENDPOINT'),
            token=os.getenv('ZILLIZ_API_KEY'),
            collection_name=collection_name,
            dim=embedding_dimension,
            overwrite=overwrite)
    else:
        milvus_vector_store = MilvusVectorStore(collection_name=collection_name,
                                                uri=os.getenv("ZILLIZ_CLOUD_URI"),
                                                token=os.getenv("ZILLIZ_CLOUD_API_KEY"),
                                                dim=embedding_dimension,
                                                overwrite=overwrite)
    storage_context = StorageContext.from_defaults(vector_store=milvus_vector_store)
    transformations = [t for t in [node_parser, sentence_splitter] if t is not None]

    VectorStoreIndex.from_documents(documents,
                                    transformations=transformations,
                                    storage_context=storage_context,
                                    show_progress=True)


def index_github_repo(owner: str,
                      repo: str,
                      branch: str,
                      collection_name: str,
                      ignore_file_extensions: list[str] = None,
                      ignore_directories: list[str] = None,
                      ):
    if ignore_file_extensions is None:
        ignore_file_extensions = ['.jpg', '.png', '.svg', '.gitignore', '.csv', '.jar', '.json', '.lock']
    if ignore_directories is None:
        ignore_directories = ['.github', '.gitignore', '2021', '2022', 'images']
    github_client = GithubClient(os.getenv('GH_TOKEN'), verbose=True)
    documents = GithubRepositoryReader(
        github_client=github_client,
        owner=owner,
        repo=repo,
        filter_directories=(ignore_directories, EXCLUDE_FILTER_TYPE),
        filter_file_extensions=(ignore_file_extensions, EXCLUDE_FILTER_TYPE),
    ).load_data(branch=branch)
    for doc in documents:
        doc.metadata['branch'] = branch
        doc.metadata['owner'] = owner
        doc.metadata['repo'] = repo
    add_route_to_docs(documents, 'github')

    ipynb_docs = [parse_ipynb_doc(doc) for doc in documents if doc.metadata.get('file_name', '').endswith('.ipynb')]
    md_docs = [doc for doc in documents if doc.metadata.get('file_name', '').endswith('.md')]
    other_docs = [doc for doc in documents if not doc.metadata.get('file_name', '').endswith(('.ipynb', '.md'))]

    add_to_index(other_docs, collection_name=collection_name)
    add_to_index(md_docs, collection_name=collection_name, node_parser=MarkdownNodeParser())
    add_to_index(ipynb_docs, collection_name=collection_name)


def remove_base64_images(markdown_text: str) -> str:
    """
    Remove all Markdown images that embed base64-encoded data URIs.

    Args:
        markdown_text: A string containing Markdown (e.g., exported from a Jupyter notebook).

    Returns:
        The same string, but with any occurrences of:
            ![alt-text](data:image/…;base64,ENCODEDDATA)
        removed entirely.
    """
    # This regex matches:
    #  - '![' followed by any number of non-']' characters ']'  → the alt text
    #  - '(' then optional whitespace, then 'data:image/' and any chars up to a ';base64,'
    #  - then all characters up to the next ')'      → the base64 blob
    pattern = re.compile(r'!\[[^]]*]\(\s*data:image/[^)]+;base64,[^)]+\)', re.IGNORECASE)
    return pattern.sub('', markdown_text)


def parse_ipynb_doc(ipynb_doc: Document) -> Document:
    ipynb_json = json.loads(ipynb_doc.text)
    temp_ipynb = tempfile.NamedTemporaryFile(suffix='.ipynb')
    try:
        with open(temp_ipynb.name, 'w') as f_out:
            json.dump(ipynb_json, f_out)
        parsed = JupyterNotebookParser(temp_ipynb.name)
        all_cells = parsed.get_all_cells()
        parsed_text = ''.join([JupyterNotebookParser._join_source_lines(cell.get('source', ''))
                               for cell in all_cells])
        ipynb_doc.text = remove_base64_images(parsed_text)
        return ipynb_doc
    finally:
        temp_ipynb.close()


def index_slack_history(channel_ids: list[str], collection_name: str):
    earliest_date = datetime.now() - timedelta(days=90)
    slack_reader = SlackReader(earliest_date=earliest_date,
                               bot_user_id=BOT_USER_ID,
                               not_ignore_users=[AU_TOMATOR_USER_ID],
                               slack_token=os.getenv("SLACK_BOT_TOKEN"))
    print('Starting to load slack messages from the last 90 days')
    documents = slack_reader.load_data(channel_ids=channel_ids)
    add_route_to_docs(documents, 'slack')
    print('Starting to add loaded Slack messages to the index')
    add_to_index(documents, collection_name=collection_name)


def index_faq_github(collection_name: str):
    """
    Index FAQ documents from GitHub repository.

    Reads FAQ markdown files from DataTalksClub/faq repository and indexes them
    for the specified zoomcamp collection.

    Args:
        collection_name: Collection name (e.g., 'mlzoomcamp_faq_git')
    """
    from ingest.readers.faq_github_reader import FAQGithubReader, ZOOMCAMP_DIR_MAPPING

    zoomcamp_name = ZOOMCAMP_DIR_MAPPING.get(collection_name)
    if not zoomcamp_name:
        raise ValueError(f"Unknown collection: {collection_name}")

    github_reader = FAQGithubReader(github_token=os.getenv('GH_TOKEN'))
    print(f'Starting to load FAQ documents from GitHub: {zoomcamp_name}')
    documents = github_reader.load_data(zoomcamp_name=zoomcamp_name)
    add_route_to_docs(documents, 'faq')
    print(f'Loaded {len(documents)} FAQ documents, adding to index')
    add_to_index(documents, collection_name=collection_name, overwrite=True)


def index_youtube(video_ids: list[str], collection_name: str):
    yt_reader = YoutubeReader()
    documents = yt_reader.load_data(video_ids=video_ids, tokenizer=embeddings.client.tokenizer)
    print('Starting to add loaded Video transcripts to the index')
    add_to_index(documents, collection_name=collection_name)
