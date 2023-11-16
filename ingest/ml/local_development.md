steps to fill in the index locally:

1. start dockerized milvus
    ```shell
     docker compose -f milvus-standalone-docker-compose.yml up    
    ```
   
1. execute. It will be executed with `EXECUTION_ENV` env var set to `local` by default
   ```shell
   python ingest_ml.py
   ```