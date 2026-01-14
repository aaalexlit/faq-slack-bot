import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ingest.utils.index_utils import index_github_repo, \
    index_slack_history, index_faq_github, index_youtube

DE_CHANNEL_ID = 'C01FABYF2RG'
FAQ_COLLECTION_NAME = 'dezoomcamp_faq_git'


def index_course_github_repo():
    owner = 'DataTalksClub'
    repo = 'data-engineering-zoomcamp'
    branch = 'main'
    index_github_repo(owner=owner,
                      repo=repo,
                      branch=branch,
                      collection_name=FAQ_COLLECTION_NAME,
                      ignore_directories=['.github', '.gitignore', 'cohorts/2022', 'cohorts/2023', 'cohorts/2024',
                                          'images'],
                      )


def index_github_faq():
    print("Indexing FAQ from GitHub")
    index_faq_github(FAQ_COLLECTION_NAME)


def index_slack_messages():
    print("Indexing slack messages")
    channel_ids = [DE_CHANNEL_ID]
    index_slack_history(channel_ids, FAQ_COLLECTION_NAME)


def index_yt_subtitles():
    print("Indexing QA videos subtitles")
    video_ids = ['X8cEEwi8DTM']
    index_youtube(video_ids, FAQ_COLLECTION_NAME)


def fill_de_index():
    print("Updating DE info Milvus index")
    print(f"Execution environment is {os.getenv('EXECUTION_ENV', 'local')}")
    index_github_faq()  # step 1

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(index_slack_messages),
            executor.submit(index_course_github_repo),
            executor.submit(index_yt_subtitles),
        ]
        for future in as_completed(futures):
            future.result()


if __name__ == '__main__':
    fill_de_index()
