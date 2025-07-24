import os
import sys
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embedding_agent import embedding_worker
from ingestion.ingest import load_and_chunk_documents

@patch("redis.Redis")
@patch("langchain.vectorstores.FAISS")
def test_embedding_worker(mock_faiss, mock_redis):
    # Mock Redis
    mock_redis_instance = MagicMock()
    mock_redis.return_value = mock_redis_instance
    mock_redis_instance.brpop.return_value = (
        b"ingest:chunks",
        json.dumps({
            "text": "test content",
            "repo_id": "test_repo",
            "file_path": "test_file.py",
            "chunk_index": 0,
        }).encode("utf-8"),
    )

    # Mock FAISS
    mock_faiss_instance = MagicMock()
    mock_faiss.return_value = mock_faiss_instance

    # Run the worker in a separate thread
    import threading
    worker_thread = threading.Thread(target=embedding_worker)
    worker_thread.daemon = True
    worker_thread.start()

    # Wait for the worker to process the item
    import time
    time.sleep(1)

    # Assert that the document was added to the vector store
    mock_faiss_instance.add_documents.assert_called_once()

@patch("redis.Redis")
def test_load_and_chunk_documents(mock_redis):
    # Mock Redis
    mock_redis_instance = MagicMock()
    mock_redis.return_value = mock_redis_instance

    # Create a dummy repo
    repo_path = "/tmp/test_load_and_chunk_documents"
    os.makedirs(repo_path, exist_ok=True)
    with open(os.path.join(repo_path, "test.py"), "w") as f:
        f.write("test content")

    # Run the function
    load_and_chunk_documents(repo_path, "test_repo")

    # Assert that the chunk was pushed to Redis
    mock_redis_instance.lpush.assert_called_once()