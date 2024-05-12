import os

from prefect import flow, task

from ingest.utils.index_utils import index_spreadsheet, index_github_repo, \
    index_slack_history, index_faq

SLACK_CHANNEL_ID = 'C02R98X7DS9'
COLLECTION_NAME = 'mlopszoomcamp'


@task(name="Index course github repo")
def index_course_github_repo():
    owner = 'DataTalksClub'
    repo = 'mlops-zoomcamp'
    branch = 'main'
    index_github_repo(owner=owner,
                      repo=repo,
                      branch=branch,
                      collection_name=COLLECTION_NAME,
                      ignore_directories=['.github', '.gitignore', 'cohorts/2022', 'cohorts/2023', 'images'],
                      )


@task(name="Index FAQ Google Document")
def index_google_doc():
    document_ids = ["12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0"]
    print('Loading google doc...')
    index_faq(document_ids, COLLECTION_NAME)


@task(name="Index course schedule")
def index_course_schedule():
    url = (
        'https://docs.google.com/spreadsheets/d/e/2PACX-1vRNTwA0Of1lyprYpn2YxU-'
        'l0gvNeq-up7g7ITB42nPf2gT9Qd3PTzqTmkjAZjk1s__r7D99CsJfcZEO/pubhtml')
    title = 'MLOps Zoomcamp 2023 Deadlines'
    index_spreadsheet(url, title, COLLECTION_NAME)


@task(name="Index project evaluation criteria")
def index_evaluation_criteria():
    url = ('https://docs.google.com/spreadsheets/d/e/2PACX'
           '-1vQCwqAtkjl07MTW-SxWUK9GUvMQ3Pv_fF8UadcuIYLgHa0PlNu9BRWtfLgivI8xSCncQs82HDwGXSm3/pubhtml')
    title = 'ML Zoomcamp project evaluation criteria : Project criteria'
    index_spreadsheet(url, title, COLLECTION_NAME)


@task(name="Index slack messages")
def index_slack_messages():
    channel_ids = [SLACK_CHANNEL_ID]
    index_slack_history(channel_ids, COLLECTION_NAME)


@flow(name="Update MLOps info Milvus index", log_prints=True)
def fill_mlops_index():
    print(f"Execution environment is {os.getenv('EXECUTION_ENV', 'local')}")
    index_google_doc()
    index_slack_messages.submit(wait_for=[index_google_doc])
    index_course_schedule.submit(wait_for=[index_google_doc])
    # index_evaluation_criteria.submit(wait_for=[index_google_doc])
    index_course_github_repo.submit(wait_for=[index_google_doc])


if __name__ == '__main__':
    fill_mlops_index()
