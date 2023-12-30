import os
from datetime import datetime

from prefect import flow, task

from ingest.utils.index_utils import index_spreadsheet, index_github_repo, \
    index_slack_history, index_faq

ML_CHANNEL_ID = 'C0288NJ5XSA'
FAQ_COLLECTION_NAME = 'mlzoomcamp_faq_git'


@task(name="Index course github repo")
def index_course_github_repo():
    owner = 'DataTalksClub'
    repo = 'machine-learning-zoomcamp'
    branch = 'master'
    index_github_repo(owner=owner,
                      repo=repo,
                      branch=branch,
                      collection_name=FAQ_COLLECTION_NAME)


@task(name="Index book github repo")
def index_book_github_repo():
    owner = 'alexeygrigorev'
    repo = 'mlbookcamp-code'
    branch = 'master'
    ignore_directories = ['.github', 'course-zoomcamp', 'images', 'util']
    index_github_repo(owner=owner,
                      repo=repo,
                      branch=branch,
                      ignore_directories=ignore_directories,
                      collection_name=FAQ_COLLECTION_NAME)


@task(name="Index FAQ Google Document")
def index_google_doc():
    document_ids = ["1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8"]
    print('Loading google doc...')
    index_faq(document_ids, FAQ_COLLECTION_NAME, question_heading_style_num=3)


@task(name="Index course schedule")
def index_course_schedule():
    url = ('https://docs.google.com/spreadsheets/d/e/2PACX'
           '-1vSkEwMv5OKwCdPfW6LgqQvKk48dZjPcFDrjDstBqZfq38UPadh0Nws1b57qOVYwzAjSufKnVf7umGWH/pubhtml')
    title = 'ML Zoomcamp 2023 syllabus and deadlines'
    index_spreadsheet(url, title, FAQ_COLLECTION_NAME)


@task(name="Index project evaluation criteria")
def index_evaluation_criteria():
    url = ('https://docs.google.com/spreadsheets/d/e/2PACX'
           '-1vQCwqAtkjl07MTW-SxWUK9GUvMQ3Pv_fF8UadcuIYLgHa0PlNu9BRWtfLgivI8xSCncQs82HDwGXSm3/pubhtml')
    title = 'ML Zoomcamp project evaluation criteria : Project criteria'
    index_spreadsheet(url, title, FAQ_COLLECTION_NAME)


@task(name="Index slack messages")
def index_slack_messages():
    earliest_date = datetime(2023, 8, 1)
    channel_ids = [ML_CHANNEL_ID]
    index_slack_history(channel_ids, earliest_date, FAQ_COLLECTION_NAME)


@flow(name="Update ML info Milvus index", log_prints=True)
def fill_ml_index():
    print(f"Execution environment is {os.getenv('EXECUTION_ENV', 'local')}")
    index_google_doc()
    index_slack_messages.submit(wait_for=[index_google_doc])
    index_course_schedule.submit(wait_for=[index_google_doc])
    index_evaluation_criteria.submit(wait_for=[index_google_doc])
    index_course_github_repo.submit(wait_for=[index_google_doc])
    index_book_github_repo.submit(wait_for=[index_google_doc])


if __name__ == '__main__':
    fill_ml_index()
