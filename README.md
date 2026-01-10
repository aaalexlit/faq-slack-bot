# FAQ Slack Bot for DataTalks.Club Zoomcamps

An LLM-powered QA Slack chatbot that answers questions related to [DataTalks.Club](https://datatalks.club/) Zoomcamps (ML, DE, MLOps, and LLM courses). The bot uses RAG (Retrieval Augmented Generation) with vector search over indexed course materials to provide accurate, context-aware answers to student questions.

## What Does This Bot Do?

The bot automatically responds to questions in Slack channels dedicated to DataTalks.Club courses by:

- **Searching indexed course materials** including FAQ documents, Slack history, GitHub repositories, and YouTube subtitles
- **Using vector similarity search** to find the most relevant content for each question
- **Re-ranking results** with Cohere's re-ranking model to improve relevance
- **Generating contextual answers** using GPT-4o-mini based on retrieved course materials
- **Providing source references** with links back to original materials (Slack messages, docs, GitHub files, videos)
- **Learning from feedback** through upvote/downvote reactions tracked via LangSmith
- **Prioritizing recent content** with time-weighted ranking to surface up-to-date information

The bot maintains separate query engines for each Zoomcamp and routes questions based on the Slack channel ID.

For a detailed explanation of how the bot works, including architecture and evaluation results, see [this comprehensive report](https://api.wandb.ai/links/aaalex-lit/ii6tpid4).

## Architecture Overview

The system consists of two main components:

### 1. Slack Bot (`slack_bot/`)
- Runs as a Slack app using Socket Mode with the Slack Bolt framework
- Maintains 4 separate query engines (ML, DE, MLOps, LLM Zoomcamps)
- Uses LlamaIndex with Milvus/Zilliz vector store and HuggingFace embeddings
- Implements time-weighted post-processing to prioritize recent Slack messages (applied first)
- Implements Cohere re-ranking to improve result relevance (20 retrieved → 10 after time-weighting → 4 after re-ranking)

### 2. Ingestion Pipeline (`ingest/`)
- Indexes data from multiple sources: Google Docs (FAQs), Slack history, GitHub repos, YouTube subtitles
- Uses embeddings cache (Upstash Redis) to reduce costs
- Runs on schedule via GitHub Actions or can be triggered manually
- Supports both local (Milvus) and production (Zilliz Cloud) vector stores

## Getting Started

### Prerequisites

- Python 3.10
- Conda (recommended) or virtualenv
- Docker and Docker Compose (for local Milvus instance)
- Slack workspace with bot permissions
- Required API keys (see Environment Variables section)

### Quick Start

#### 1. Running the Bot Locally

1. **Set up the environment:**
   ```bash
   conda create --name slack-bot python=3.10
   conda activate slack-bot
   cd slack_bot
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   - In the `slack_bot/` directory, rename `dev.env` to `.env`
   - Fill in all required values (see [Environment Variables](#environment-variables))

3. **Start local Milvus instance** (in a separate terminal):
   ```bash
   cd ingest/local_milvus
   docker compose up
   ```

4. **Run ingestion to populate the index** (in another terminal)

   **Important:** Before running the bot, you must populate the vector store with course data. See the [Running Ingestion Locally](#2-running-ingestion-locally) section below for detailed instructions.

5. **Start the bot** (in the original terminal or a new one):
   ```bash
   cd slack_bot
   source .env
   python main.py
   ```

For more details, see [`slack_bot/README.md`](slack_bot/README.md).

#### 2. Running Ingestion Locally

1. **Start local Milvus** (in a separate terminal):
   ```bash
   cd ingest/local_milvus
   docker compose up
   ```

2. **Configure environment variables:**
   - In the `ingest/` directory, rename `dev.env` to `.env`
   - Fill in all required values

3. **Run ingestion for specific courses** (in another terminal, from the project root):
   ```bash
   source ingest/.env
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   python ingest/ml/ingest_ml.py      # for ML Zoomcamp
   python ingest/de/ingest_de.py      # for DE Zoomcamp
   python ingest/llm/ingest_llm.py    # for LLM Zoomcamp
   python ingest/mlops/ingest_mlops.py # for MLOps Zoomcamp
   ```

For more details, see [`ingest/local_development.md`](ingest/local_development.md).

## Environment Variables

> ⚠️ **Security Note:** Never commit `.env` files or expose API keys in your code. Keep all credentials private and ensure `.env` is listed in your `.gitignore`.

### Required for Slack Bot
- `SLACK_BOT_TOKEN` - Bot token for Slack app
- `SLACK_APP_TOKEN` - App-level token for Socket Mode
- `OPENAI_API_KEY` - OpenAI API key for GPT-4o-mini
- `COHERE_API_KEY` - Cohere API key for result re-ranking
- `ZILLIZ_CLOUD_URI`, `ZILLIZ_CLOUD_API_KEY` - For ML/DE Zoomcamps
- `ZILLIZ_PUBLIC_ENDPOINT`, `ZILLIZ_API_KEY` - For MLOps/LLM Zoomcamps
- `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` - For LangSmith logging
- `LOCAL_MILVUS` - Set to `true` for local development

### Required for Ingestion
- All of the above, plus:
- `GH_TOKEN` - GitHub personal access token for API access
- `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` - For embeddings cache
- `EMBEDDING_CACHE_NAMESPACE` - Set per course to avoid cache collisions
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to service account key for Google Docs API
- `EXECUTION_ENV` - Set to `local` for local dev, `zilliz-cluster` for MLOps/LLM production, or leave empty for ML/DE production

## Deployment

### Slack Bot
- Deployed to [Fly.io](https://fly.io/)
- Docker image: `aaalexlit/faq-slack-bot`
- Automated via GitHub Actions:
  - `.github/workflows/docker-image.yml` - Builds Docker image
  - `.github/workflows/fly-deploy.yml` - Deploys to Fly.io

### Ingestion Pipeline
- Runs on schedule via GitHub Actions (see `.github/workflows/schedule-ingest-*.yml`)
- Docker image: `aaalexlit/zoomcamp-faq-ingest`
- Can be triggered manually via `.github/workflows/run-ingest.yml`

## Technology Stack

- **Programming Language**: Python 3.10
- **LLM Framework**: LlamaIndex
- **Vector Database**: Milvus (local), Zilliz Cloud (production)
- **Embeddings**: HuggingFace BAAI/bge-base-en-v1.5
- **Re-ranking**: Cohere Rerank API
- **LLM**: OpenAI GPT-4o-mini
- **Slack Integration**: Slack Bolt (Socket Mode)
- **Observability**: LangSmith
- **Embeddings Cache**: Upstash Redis

## Project Structure

```
├── slack_bot/          # Slack bot application
│   ├── main.py         # Entry point
│   └── README.md       # Bot setup instructions
├── ingest/             # Data ingestion pipeline
│   ├── ml/             # ML Zoomcamp ingestion
│   ├── de/             # DE Zoomcamp ingestion
│   ├── llm/            # LLM Zoomcamp ingestion
│   ├── mlops/          # MLOps Zoomcamp ingestion
│   ├── readers/        # Custom data readers
│   ├── utils/          # Shared utilities
│   └── local_milvus/   # Docker Compose for local Milvus
├── .github/workflows/  # CI/CD workflows
└── CLAUDE.md           # Development guidelines for Claude Code
```

## Contributing

For development guidelines and detailed implementation notes, see [`CLAUDE.md`](CLAUDE.md).

## License

This project is part of [DataTalks.Club](https://datatalks.club/) community resources.