import json
import os
import time

from prefect.blocks.system import Secret
from prefect_gcp import GcpCredentials


def create_gcp_creds_block():
    block_name = "google-drive-creds"
    try:
        GcpCredentials.load(block_name)
    except ValueError:
        with open("../keys/service_account_key.json", 'r') as f_in:
            service_account_info_str = f_in.read()

        service_account_info = json.loads(service_account_info_str)

        GcpCredentials(
            service_account_info=service_account_info
        ).save(block_name)


def create_secret_block(block_name: str, env_var_name: str) -> None:
    try:
        Secret.load(block_name)
    except ValueError:
        Secret(value=os.getenv(env_var_name)).save(name=block_name)


def create_pinecone_secrets():
    create_secret_block('pinecone-api-key', 'PINECONE_API_KEY')
    time.sleep(10)
    create_secret_block('pinecone-env', 'PINECONE_ENV')


def create_zilliz_secrets():
    create_secret_block('zilliz-cloud-uri', 'ZILLIZ_CLOUD_URI')
    time.sleep(10)
    create_secret_block('zilliz-cloud-api-key', 'ZILLIZ_CLOUD_API_KEY')


def create_slack_secrets():
    create_secret_block('slack-bot-token', 'SLACK_BOT_TOKEN')


def create_github_secrets():
    create_secret_block('github-token', 'GITHUB_TOKEN')


if __name__ == '__main__':
    create_gcp_creds_block()
    time.sleep(10)
    create_pinecone_secrets()
    time.sleep(10)
    create_zilliz_secrets()
    time.sleep(10)
    create_slack_secrets()
    time.sleep(10)
    create_github_secrets()
