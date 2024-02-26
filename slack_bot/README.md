# Running the bot locally

1. Rename [dev.env](../dev.env) to `.env` and set all the required variables

1. Run ingestion with local milvus following [local_development.md](../ingest/local_development.md)

1. Run [main.py](main.py)

    ```shell
    source .env
    python main.py
    ```
    In Pycharm IDE use a provided run configuration [run_bot_local_ws.run.xml](../.run/run_bot_local_ws.run.xml)
