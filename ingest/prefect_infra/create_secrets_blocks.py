import json
import os
import time

from prefect_gcp import GcpCredentials
from prefect.blocks.system import Secret


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


def create_pinecone_secrets():
    pinecone_api_key_block_name = 'pinecone-api-key'
    pinecone_env_block_name = 'pinecone-env'
    try:
        Secret().load(pinecone_api_key_block_name)
    except ValueError:
        Secret(value=os.getenv('PINECONE_API_KEY')).save(name=pinecone_api_key_block_name)
    time.sleep(10)
    try:
        Secret().load(pinecone_env_block_name)
    except ValueError:
        Secret(value=os.getenv('PINECONE_ENV')).save(name=pinecone_env_block_name)


if __name__ == '__main__':
    create_gcp_creds_block()
    time.sleep(10)
    create_pinecone_secrets()
