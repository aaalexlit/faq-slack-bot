# Run ingestion locally for ML and DE Zoomcamps

Steps to fill in the index locally:

1. start dockerized [Milvus](https://milvus.io/) from [local_milvus](local_milvus) folder
    ```shell
     cd ingest/local_milvus
     docker compose up    
    ```
   
1. Rename [dev.env](../dev.env) to `.env` and set all the required variables

1. execute ingestion script [ingest_ml.py](ml/ingest_ml.py) (for ML zoomcamp data) 
or [ingest_de.py](de/ingest_de.py) (for DE zoomcamp data).  
It will be executed with `EXECUTION_ENV` env var set to `local` by default
   ```shell
   export PYTHONPATH="${PYTHONPATH}:$(pwd)" 
   python ingest/ml/ingest_ml.py
   ```
   
   If you're using Pycharm IDE there are run configurations available:  
   [ingest_de](../.run/ingest_de.run.xml)  
   [ingest_ml](../.run/ingest_ml.run.xml)