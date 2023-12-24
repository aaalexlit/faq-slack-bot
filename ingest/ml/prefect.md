# Run the ingestion for ML with prefect deployments

## Execute ingestion

Currently, indexing is scheduled to execute every 24 hours at 23 CET.
Ad-hoc executions can be run from the [Prefect Cloud UI](https://app.prefect.cloud/)
by launching the corresponding deployment.
It's also possible to run it from the command line
```shell
prefect deployment run 'Update ML info Milvus index/fill-index-zilliz'
```

## Change the properties of the current deployment 

Depending on the nature of the changes, after modifying the code or 
[prefect.yaml](prefect.yaml) re-create the deployment by running

```shell
prefect deploy --all
```

## Setup prefect from scratch

Login to prefect cloud:

```shell
prefect cloud login
```

Create the required blocks:

```shell
python ingest/prefect_infra/create_secrets_blocks.py
```

Create work pool

```shell
prefect work-pool create --type docker zoomcamp-faq-bot
```

Run the following command in this new terminal to start the worker:

```shell
prefect worker start --pool zoomcamp-faq-bot
```

Create all the deployments from [prefect.yaml](prefect.yaml) file

```shell
prefect deploy --all
```
