# Run ingestion locally for ML and DE Zoomcamps

Steps to fill in the index locally:

1. start dockerized [Milvus](https://milvus.io/) from [local_milvus](local_milvus) folder
    ```shell
     cd ingest/local_milvus
     docker compose up    
    ```
   
1. execute ingestion script [ingest_ml.py](ml/ingest_ml.py). 
It will be executed with `EXECUTION_ENV` env var set to `local` by default
   ```shell
   export PYTHONPATH="${PYTHONPATH}:$(pwd)" 
   python ingest/ml/ingest_ml.py
   ```