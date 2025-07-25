import pytest
import os
import sys
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embedding_agent import embedding_worker
from ingestion.ingest import load_and_chunk_documents

@patch("embedding_agent.vectordb")
def test_embedding_worker(mock_vectordb):
    # Mock Redis
    mock_redis_instance = MagicMock()
    mock_redis_instance.brpop.side_effect = [
        (
            b"ingest:chunks",
            json.dumps({
                "text": "test content",
                "repo_id": "test_repo",
                "file_path": "test_file.py",
                "chunk_index": 0,
            }).encode("utf-8"),
        ),
        StopIteration,
    ]

    # Run the worker
    embedding_worker(mock_redis_instance)

    # Assert that the document was added to the vector store
    mock_vectordb.add_documents.assert_called_once()

@patch("ingestion.ingest.redis.Redis")
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

if __name__ == "__main__":
    pytest.main()