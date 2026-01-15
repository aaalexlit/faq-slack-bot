import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ingest.utils.index_utils import index_github_repo, \
    index_slack_history, index_faq_github

ML_CHANNEL_ID = 'C0288NJ5XSA'
FAQ_COLLECTION_NAME = 'mlzoomcamp_faq_git'


def index_course_github_repo():
    print("Indexing course github repo")
    owner = 'DataTalksClub'
    repo = 'machine-learning-zoomcamp'
    branch = 'master'
    index_github_repo(owner=owner,
                      repo=repo,
                      branch=branch,
                      collection_name=FAQ_COLLECTION_NAME)


def index_book_github_repo():
    print("Indexing book github repo")
    owner = 'alexeygrigorev'
    repo = 'mlbookcamp-code'
    branch = 'master'
    ignore_directories = ['.github', 'course-zoomcamp', 'images', 'util']
    index_github_repo(owner=owner,
                      repo=repo,
                      branch=branch,
                      ignore_directories=ignore_directories,
                      collection_name=FAQ_COLLECTION_NAME)


def index_github_faq():
    print("Indexing FAQ from GitHub")
    index_faq_github(FAQ_COLLECTION_NAME)


def index_slack_messages():
    print("Indexing slack messages")
    channel_ids = [ML_CHANNEL_ID]
    index_slack_history(channel_ids, FAQ_COLLECTION_NAME)


def fill_ml_index():
    print("Updating ML info Milvus index")
    print(f"Execution environment is {os.getenv('EXECUTION_ENV', 'local')}")
    # 1) do the GitHub FAQ indexing first
    index_github_faq()
    # 2) now run the other tasks in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(index_slack_messages),
            executor.submit(index_course_github_repo),
            executor.submit(index_book_github_repo),
        ]
        # optional: wait for both to finish (and raise if either errors)
        for future in as_completed(futures):
            future.result()


if __name__ == '__main__':
    fill_ml_index()
