# Running the bot locally

1. re-create separate conda environment using [slack_bot/requirements.txt](../slack_bot/requirements.txt)
    ```shell
    conda activate base
    conda remove --name slack-bot --all
    conda create --name slack-bot python=3.10
    conda activate slack-bot
    cd slack_bot
    pip install -r requirements.txt
    ```
1. Rename [dev.env](../dev.env) to `.env` and set all the required variables

1. Run ingestion with local milvus following [local_development.md](../ingest/local_development.md)

1. Run [main.py](main.py)

    ```shell
    source .env
    python main.py
    ```
    In Pycharm IDE use a provided run configuration [run_bot_local_ws.run.xml](../.run/run_bot_local_ws.run.xml)
