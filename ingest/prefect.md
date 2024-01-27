# Run the ingestion for ML with prefect deployments

## Execute ingestion

Currently, indexing is scheduled to execute:
- Daily at 00:00 CET for **DE Zoomcamp** documents
- Weekly at 23:00 CET on Monday for **ML Zoomcamp** documents

Before running any execution make sure the worker is started:
```shell
prefect worker start --pool zoomcamp-faq-bot
```

Ad-hoc executions can be run from the [Prefect Cloud UI](https://app.prefect.cloud/)
by launching the corresponding deployment.  


It's also possible to run it from the command line:

### Run ingestion deployment for ML
```shell
prefect deployment run 'Update ML info Milvus index/fill-index-zilliz-ml'
```

### Run ingestion deployment for DE
```shell
prefect deployment run 'Update DE info Milvus index/fill-index-zilliz-de'
```

## Change the properties of a deployment 
### Bulk
Depending on the nature of the changes, after modifying the code or 
[prefect.yaml](../prefect.yaml) re-create both deployments by running

```shell
prefect deploy --all
```
### Individual
Alternatively it can be done per deployment if the changes are not affecting both  
**re-create deployment for ML ingestion**
```shell
prefect deploy --name fill-index-zilliz-ml
```
**re-create deployment for DE ingestion**
```shell
prefect deploy --name fill-index-zilliz-de
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

Create all the deployments from [prefect.yaml](../prefect.yaml) file

```shell
prefect deploy --all
```

Run the ingestion by executing created deployments following the
instructions above.