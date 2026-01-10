import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ingest.utils.index_utils import index_github_repo, \
    index_slack_history, index_faq_github, index_youtube

SLACK_CHANNEL_ID = 'C06TEGTGM3J'
COLLECTION_NAME = 'llmzoomcamp'


def index_course_github_repo():
    print("Indexing course github repo")
    owner = 'DataTalksClub'
    repo = 'llm-zoomcamp'
    branch = 'main'
    index_github_repo(owner=owner,
                      repo=repo,
                      branch=branch,
                      collection_name=COLLECTION_NAME,
                      ignore_directories=['.github', '.gitignore', 'images', 'cohorts/2024'],
                      )


def index_faq_documents():
    print("Indexing FAQ from GitHub")
    index_faq_github('llm-zoomcamp', COLLECTION_NAME)


def index_slack_messages():
    print("Indexing slack messages")
    channel_ids = [SLACK_CHANNEL_ID]
    index_slack_history(channel_ids, COLLECTION_NAME)

def index_yt_subtitles():
    print("Indexing QA videos subtitles")
    video_ids = ['GH3lrOsU3AU', 'FgnelhEJFj0', '8lgiOLMMKcY']
    index_youtube(video_ids, COLLECTION_NAME)

def fill_llm_index():
    print("Updating LLM info Milvus index")
    print(f"Execution environment is {os.getenv('EXECUTION_ENV', 'local')}")
    # 1) do the FAQ indexing first
    index_faq_documents()

    # 2) run the other in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(index_slack_messages),
            executor.submit(index_course_github_repo),
            executor.submit(index_yt_subtitles),
        ]
        # wait for both to finish (and raise if either errors)
        for future in as_completed(futures):
            future.result()


if __name__ == '__main__':
    fill_llm_index()
