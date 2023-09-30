Build docker container
```shell
docker build -t aaalexlit/faq-slack-bot:0.0.x . 
```

Run it

```shell
docker run -d --env-file .env aaalexlit/faq-slack-bot:0.0.x
```

Push to hub

```shell
docker push aaalexlit/faq-slack-bot:0.0.x
```