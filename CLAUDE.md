# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM-powered QA Slack chatbot that answers questions related to DataTalks.Club Zoomcamps (ML, DE, MLOps, and LLM courses). The bot uses RAG (Retrieval Augmented Generation) with vector search over indexed course materials including FAQ documents, Slack history, GitHub repositories, and YouTube subtitles.

For detailed explanation of how the bot works, see [this report](https://api.wandb.ai/links/aaalex-lit/ii6tpid4).

## Architecture

The system consists of two main components:

### 1. Slack Bot (`slack_bot/`)
- **Entry point**: `slack_bot/main.py`
- Runs as a Slack app using Socket Mode (Slack Bolt framework)
- Maintains 4 separate query engines, one per Zoomcamp:
  - ML Zoomcamp (`mlzoomcamp_faq_git` collection)
  - DE Zoomcamp (`dezoomcamp_faq_git` collection)
  - MLOps Zoomcamp (`mlopszoomcamp` collection)
  - LLM Zoomcamp (`llmzoomcamp` collection)
- Routes questions to the appropriate engine based on channel ID (see `slack_bot/main.py:34-39`)
- Uses LlamaIndex with Milvus/Zilliz vector store and HuggingFace embeddings (`BAAI/bge-base-en-v1.5`)
- Uses OpenAI GPT-4o-mini (`gpt-4o-mini-2024-07-18`) for response generation
- Implements feedback collection (upvote/downvote) tracked via LangSmith
- Applies `TimeWeightedPostprocessor` to prioritize recent content (90-day window for Slack messages)

### 2. Ingestion Pipeline (`ingest/`)
- **Entry points**: Course-specific scripts in `ingest/{ml,de,mlops,llm}/ingest_*.py`
- **Shared utilities**: `ingest/utils/index_utils.py`
- **Custom readers**: `ingest/readers/` for Slack, Google Docs, YouTube
- Indexes data from multiple sources:
  - Google Docs (FAQ documents) via custom `FAQGoogleDocsReader`
  - Slack history (90 days) via custom `SlackReader`
  - GitHub repositories (course repos) via `GithubRepositoryReader`
  - YouTube videos (subtitles) via custom `YoutubeReader`
- Uses embeddings cache backed by Upstash Redis to reduce costs
- Executes indexing in parallel using `ThreadPoolExecutor` where possible
- Supports local (Milvus) and production (Zilliz Cloud) vector stores via `EXECUTION_ENV` environment variable
- Different node parsers for different content types:
  - Jupyter notebooks: custom parser that strips base64 images
  - Markdown files: `MarkdownNodeParser`
  - Other files: `SentenceSplitter` (512 chunk size, 50 overlap)

## Common Development Commands

### Running the Slack Bot Locally

1. Set up environment:
   ```bash
   conda create --name slack-bot python=3.10
   conda activate slack-bot
   cd slack_bot
   pip install -r requirements.txt
   ```

2. Configure environment variables (rename `dev.env` to `.env` and fill in values)

3. Start local Milvus instance:
   ```bash
   cd ingest/local_milvus
   docker compose up
   ```

4. Run the bot:
   ```bash
   source .env
   python main.py
   ```

### Running Ingestion Locally

1. Start local Milvus:
   ```bash
   cd ingest/local_milvus
   docker compose up
   ```

2. Configure environment variables (rename `dev.env` to `.env` and fill in values)

3. Run ingestion for a specific course:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   python ingest/ml/ingest_ml.py      # for ML Zoomcamp
   python ingest/de/ingest_de.py      # for DE Zoomcamp
   python ingest/llm/ingest_llm.py    # for LLM Zoomcamp
   python ingest/mlops/ingest_mlops.py # for MLOps Zoomcamp
   ```

By default, scripts use `EXECUTION_ENV=local`.

## Deployment

### Slack Bot
- Deployed to Fly.io
- Docker image: `aaalexlit/faq-slack-bot`
- Workflow: `.github/workflows/docker-image.yml` builds image → `.github/workflows/fly-deploy.yml` deploys

### Ingestion
- Runs via GitHub Actions on schedule (see `.github/workflows/schedule-ingest-*.yml`)
- Docker image: `aaalexlit/zoomcamp-faq-ingest`
- Manual trigger available via `.github/workflows/run-ingest.yml`
- Ingestion runs with production environment (`EXECUTION_ENV=zilliz-cluster` for MLOps/LLM, default for ML/DE)

## Important Implementation Details

### Channel-Based Routing
The bot determines which course to query based on Slack channel ID:
- DE channels: `['C01FABYF2RG', 'C06CBSE16JC', 'C06BZJX8PSP']`
- ML channels: `['C0288NJ5XSA', 'C05C3SGMLBB', 'C05DTQECY66']`
- MLOps channels: `['C02R98X7DS9', 'C06C1N46CQ1', 'C0735558X52']`
- LLM channels: `['C079QE5NAMP', 'C078X7REVN3', 'C06TEGTGM3J']`

When adding a new channel, update the appropriate list in `slack_bot/main.py:34-39`.

### Prompt Engineering
Each course has a customized prompt template (see `get_prompt_template()` in `slack_bot/main.py:369-415`) that includes:
- Course name, cohort year, and start date
- Current date for temporal awareness
- Guidelines for answering student questions
- Instructions for link formatting (Slack-style: `<url|text>`)

When updating cohorts, modify the query engine initialization in `slack_bot/main.py:486-504`.

### Vector Store Configuration
Different collections use different Zilliz clusters:
- ML/DE: Use `ZILLIZ_CLOUD_URI` and `ZILLIZ_CLOUD_API_KEY`
- MLOps/LLM: Use `ZILLIZ_PUBLIC_ENDPOINT` and `ZILLIZ_API_KEY`

This is handled in both `slack_bot/main.py:422-440` and `ingest/utils/index_utils.py:81-98`.

### Metadata and Source Attribution
Source nodes include metadata for generating reference links:
- **Slack**: `channel`, `thread_ts` → formatted as `https://datatalks-club.slack.com/archives/{channel}/p{thread_ts}`
- **Google Docs**: `source` (URL), `title`, `section_name`
- **GitHub**: `owner`, `repo`, `branch`, `file_path` → formatted as GitHub blob URL
- **YouTube**: `yt_link`, `yt_title`

See `links_to_source_nodes()` in `slack_bot/main.py:275-308` for formatting logic.

### Ignored Content During Indexing
When indexing GitHub repos, certain directories are excluded (older cohorts, images, etc.):
- Default ignore: `.github`, `.gitignore`, `2021`, `2022`, `images`
- Course-specific: See individual `ingest_*.py` files for additional excludes

File extensions ignored: `.jpg`, `.png`, `.svg`, `.gitignore`, `.csv`, `.jar`, `.json`, `.lock`

## Environment Variables

Required for Slack bot:
- `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`
- `OPENAI_API_KEY`
- `ZILLIZ_CLOUD_URI`, `ZILLIZ_CLOUD_API_KEY` (for ML/DE)
- `ZILLIZ_PUBLIC_ENDPOINT`, `ZILLIZ_API_KEY` (for MLOps/LLM)
- `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` (for LangSmith logging)
- `LOCAL_MILVUS` (set for local development)

Required for ingestion:
- All of the above
- `GH_TOKEN` (for GitHub API access)
- `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` (for embeddings cache)
- `EMBEDDING_CACHE_NAMESPACE` (set per course)
- `GOOGLE_APPLICATION_CREDENTIALS` (path to service account key for Google Docs)
- `EXECUTION_ENV` (`local`, `zilliz-cluster`, or empty for production)