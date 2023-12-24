# Execute indexing
## For ML Zoomcamp
At the moment the indexing is scheduled to execute with [Prefect Cloud](https://app.prefect.cloud/)
via deployments every 24 hours at 23 CET

Steps to change/run the deployment are described in [prefect.md](ml/prefect.md)

## For MLOps Zoomcamp

Execute [ingest.py](mlops/ingest.py)
```shell
python ingest.py
```

# Setup Prefect

To run any ingestion, Prefect needs to be set up, 
as the code relies on secrets stored in Prefect blocks.

## Create a new profile to use with the cloud and use it (Optional)

```bash
prefect profile create cloud
prefect profile use cloud
```

## Log in to prefect cloud either though browser or using the API key
```bash
prefect cloud login
```

Create the required prefect blocks. Make sure to set up corresponding environment
variables.

```shell
python ingest/prefect_infra/create_secrets_blocks.py
```