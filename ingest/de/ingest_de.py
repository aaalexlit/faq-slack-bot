import os

from prefect import flow, task

from ingest.utils.index_utils import index_spreadsheet, index_github_repo, \
    index_slack_history, index_faq, index_youtube

DE_CHANNEL_ID = 'C01FABYF2RG'
FAQ_COLLECTION_NAME = 'dezoomcamp_faq_git'

os.environ['PREFECT_LOGGING_EXTRA_LOGGERS'] = 'llama-index-core'


@task(name="Index course github repo")
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


@task(name="Index risingwave zoomcamp github repo")
def index_risingwave_zoomcamp_github_repo():
    owner = 'risingwavelabs'
    repo = 'risingwave-data-talks-workshop-2024-03-04'
    branch = 'main'
    index_github_repo(owner=owner,
                      repo=repo,
                      branch=branch,
                      collection_name=FAQ_COLLECTION_NAME,
                      ignore_directories=['assets', 'data'],
                      ignore_file_extensions=['.gitignore', '.parquet', '.csv'])


@task(name="Index mage zoomcamp github repo")
def index_mage_zoomcamp_github_repo():
    owner = 'mage-ai'
    repo = 'mage-zoomcamp'
    branch = 'solutions'
    index_github_repo(owner=owner,
                      repo=repo,
                      branch=branch,
                      collection_name=FAQ_COLLECTION_NAME,
                      ignore_directories=[],
                      ignore_file_extensions=['.gitignore'])


@task(name="Index FAQ Google Document")
def index_google_doc():
    document_ids = ["19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw"]
    print('Loading google doc...')
    index_faq(document_ids, FAQ_COLLECTION_NAME)


@task(name="Index course schedule")
def index_course_schedule():
    url = (
        'https://docs.google.com/spreadsheets/d/e/2PACX-1vQACMLuutV5rvXg5qICuJGL-'
        'yZqIV0FBD84CxPdC5eZHf8TfzB-CJT_3Mo7U7oGVTXmSihPgQxuuoku/pubhtml')
    title = 'DE Zoomcamp 2024 syllabus and deadlines'
    index_spreadsheet(url, title, FAQ_COLLECTION_NAME)


@task(name="Index slack messages")
def index_slack_messages():
    channel_ids = [DE_CHANNEL_ID]
    index_slack_history(channel_ids, FAQ_COLLECTION_NAME)


@task(name="Index QA videos subtitles")
def index_yt_subtitles():
    video_ids = ['X8cEEwi8DTM']
    index_youtube(video_ids, FAQ_COLLECTION_NAME)


@flow(name="Update DE info Milvus index", log_prints=True)
def fill_de_index():
    print(f"Execution environment is {os.getenv('EXECUTION_ENV', 'local')}")
    index_google_doc()
    index_slack_messages.submit(wait_for=[index_google_doc])
    index_course_schedule.submit(wait_for=[index_google_doc])
    # index_evaluation_criteria.submit(wait_for=[index_google_doc])
    index_course_github_repo.submit(wait_for=[index_google_doc])
    index_yt_subtitles.submit(wait_for=[index_google_doc])


if __name__ == '__main__':
    fill_de_index()
