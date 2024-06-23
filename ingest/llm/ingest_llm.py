import os

from prefect import flow, task

from ingest.utils.index_utils import index_github_repo, \
    index_slack_history, index_faq

SLACK_CHANNEL_ID = 'C06TEGTGM3J'
COLLECTION_NAME = 'llmzoomcamp'


@task(name="Index course github repo")
def index_course_github_repo():
    owner = 'DataTalksClub'
    repo = 'llm-zoomcamp'
    branch = 'main'
    index_github_repo(owner=owner,
                      repo=repo,
                      branch=branch,
                      collection_name=COLLECTION_NAME,
                      ignore_directories=['.github', '.gitignore', 'images'],
                      )


@task(name="Index FAQ Google Document")
def index_google_doc():
    document_ids = ["1m2KexowAXTmexfC5rVTCSnaShvdUQ8Ag2IEiwBDHxN0"]
    print('Loading google doc...')
    index_faq(document_ids, COLLECTION_NAME)


@task(name="Index slack messages")
def index_slack_messages():
    channel_ids = [SLACK_CHANNEL_ID]
    index_slack_history(channel_ids, COLLECTION_NAME)


@flow(name="Update LLM info Milvus index", log_prints=True)
def fill_llm_index():
    print(f"Execution environment is {os.getenv('EXECUTION_ENV', 'local')}")
    index_google_doc()
    index_slack_messages.submit(wait_for=[index_google_doc])
    index_course_github_repo.submit(wait_for=[index_google_doc])


if __name__ == '__main__':
    fill_llm_index()
