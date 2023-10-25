# Run the ingestion for ML with prefect deployments

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

Now all the deployments can be run from the Prefect Clouf UI
It's also possible to run it from the command line
```shell
prefect deployment run 'Update ML info Milvus index/fill-index-zilliz' -p param_name=param_value
```