# Run ingestion locally

Follow these steps to populate the index on your local machine:

1. Start the dockerised [Milvus](https://milvus.io/) instance from the [local_milvus](local_milvus) folder:
   ```shell
   cd ingest/local_milvus
   docker compose up
   ```
   
1. Rename [dev.env](../dev.env) to `.env` and set all required variables

1. Execute the relevant ingestion script:

   - [ingest_ml.py](ml/ingest_ml.py) for ML Zoomcamp data  
   - [ingest_de.py](de/ingest_de.py) for DE Zoomcamp data  
   - [ingest_llm.py](llm/ingest_llm.py) for LLM Zoomcamp data  
   - [ingest_mlops.py](mlops/ingest_mlops.py) for MLOps Zoomcamp data  
   By default, the scripts run with the environment variable `EXECUTION_ENV=local`.

   Example:
   ```shell
   export PYTHONPATH="${PYTHONPATH}:$(pwd)" 
   python ingest/ml/ingest_ml.py
   ```
   
If you use PyCharm, ready-made run configurations are available:
   - [ingest_de](../.run/ingest_de.run.xml)  
   - [ingest_ml](../.run/ingest_ml.run.xml)  
   - [ingest_llm](../.run/ingest_llm.run.xml)  
   - [ingest_mlops](../.run/ingest_mlops.run.xml)
