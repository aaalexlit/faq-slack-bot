Build docker container
```shell
docker build -t aaalexlit/mlops-faq:0.0.1 . 
```

Run it

```shell
docker run -d  --name mlops-faq --env-file .env aaalexlit/mlops-faq:0.0.1
```