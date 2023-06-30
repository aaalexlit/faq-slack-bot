# To run using Prefect 

## Create a new profile to use with the cloud and use it (Optional)

```bash
prefect profile create cloud
prefect profile use cloud
```

## Log in to prefect cloud either though browser or using the API key
```bash
prefect cloud login
```

## Run as usual
```shell
python ingest.py
```
