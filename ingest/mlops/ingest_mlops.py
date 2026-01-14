import os

from concurrent.futures import ThreadPoolExecutor, as_completed

from ingest.utils.index_utils import index_github_repo, \
    index_slack_history, index_faq_github

SLACK_CHANNEL_ID = 'C02R98X7DS9'
COLLECTION_NAME = 'mlopszoomcamp'


def index_course_github_repo():
    print("Indexing course github repo")
    owner = 'DataTalksClub'
    repo = 'mlops-zoomcamp'
    branch = 'main'
    index_github_repo(owner=owner,
                      repo=repo,
                      branch=branch,
                      collection_name=COLLECTION_NAME,
                      ignore_directories=['.github', '.gitignore', 'cohorts/2022', 'cohorts/2023', 'cohorts/2024',
                                          'images'],
                      )


def index_github_faq():
    print("Indexing FAQ from GitHub")
    index_faq_github(COLLECTION_NAME)


def index_slack_messages():
    print("Indexing slack messages")
    channel_ids = [SLACK_CHANNEL_ID]
    index_slack_history(channel_ids, COLLECTION_NAME)


def fill_mlops_index():
    print("Updating MLOps info Milvus index")
    print(f"Execution environment is {os.getenv('EXECUTION_ENV', 'local')}")
    index_github_faq()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(index_slack_messages),
            executor.submit(index_course_github_repo),
        ]
        # optional: wait for both to finish (and raise if either errors)
        for future in as_completed(futures):
            future.result()


if __name__ == '__main__':
    fill_mlops_index()
