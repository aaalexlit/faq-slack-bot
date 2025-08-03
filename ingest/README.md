# Indexing Execution

## GitHub Actions (Production)

The ingestion workflow is orchestrated by three main GitHub Actions:

1. **Build & Push Image**  
   Builds the ingestion Docker image and pushes it to Docker Hub.  
   [Workflow file](../.github/workflows/build-ingest-image.yml)

1. **Parameterized Ingestion Run**  
   Triggers an ingestion job with runtime-supplied parameters.  
   [Workflow file](../.github/workflows/run-ingest.yml)

1. **Course-Specific Scheduled Runs**  
   Individually scheduled runs for each ZoomCamp deployment:  
   • [LLM](../.github/workflows/schedule-ingest-llm.yml)  
   • [DE](../.github/workflows/schedule-ingest-de.yml)  
   • [ML Ops](../.github/workflows/schedule-ingest-mlops.yml)  
   • [ML](../.github/workflows/schedule-ingest-ml.yml)

## Local Execution

Instructions for running the ingestion pipeline locally are provided in  
[local_development.md](local_development.md).