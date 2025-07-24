# Repository Ingestion Agent

This project is a repository ingestion agent that can clone a git repository, split the text into chunks, generate embeddings, and store them in a FAISS vector store. It exposes a FastAPI endpoint to trigger the ingestion process and uses a Redis task queue to process requests asynchronously.

## Setup

1.  **Install the dependencies:**

    ```
    pip install -r requirements.txt
    ```

2.  **Start a Redis server.**

3.  **Start the Celery worker:**

    ```
    celery -A ingestion.tasks worker --loglevel=info
    ```

4.  **Start the FastAPI server:**

    ```
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

## Usage

To ingest a repository, send a POST request to the `/ingest` endpoint with the following JSON payload:

```json
{
  "repo_url": "https://github.com/langchain-ai/langchain.git",
  "embeddings_type": "huggingface",
  "db_path": "faiss_index"
}
```

The `embeddings_type` can be either `huggingface` or `google`.