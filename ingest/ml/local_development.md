Steps to fill in the index locally:

1. start dockerized [Milvus](https://milvus.io/) from [local_milvus](../local_milvus) folder
    ```shell
     cd ingest/local_milvus
     docker compose up    
    ```
   
1. execute ingestion script. 
It will be executed with `EXECUTION_ENV` env var set to `local` by default
   ```shell
   python ingest_ml.py
   ```