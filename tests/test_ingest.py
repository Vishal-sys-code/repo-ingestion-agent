import os
import shutil
import sys
import os
from unittest.mock import patch, MagicMock
import pytest
import redis
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.worker import enqueue_repo, ingestion_worker
from ingestion.ingest import create_vector_store_from_chunks

@pytest.fixture
def mock_redis_client():
    with patch("ingestion.worker.r", MagicMock()) as mock_redis_worker:
        with patch("ingestion.ingest.redis.Redis") as mock_redis_ingest_class:
            mock_redis_ingest_instance = MagicMock()
            mock_redis_ingest_class.return_value = mock_redis_ingest_instance
            yield (mock_redis_worker, mock_redis_ingest_instance)

def test_enqueue_repo(mock_redis_client):
    mock_redis_worker, _ = mock_redis_client
    repo_url = "https://github.com/test/repo.git"
    repo_id = "test_repo"
    enqueue_repo(repo_url, repo_id)
    expected_payload = json.dumps({"repo_url": repo_url, "repo_id": repo_id})
    mock_redis_worker.lpush.assert_called_once_with("ingest:repos", expected_payload)

@patch("ingestion.worker.Repo.clone_from")
@patch("ingestion.worker.os.walk")
def test_ingestion_worker(mock_walk, mock_clone, mock_redis_client):
    mock_redis_worker, _ = mock_redis_client
    mock_redis_worker.brpop.return_value = (None, json.dumps({"repo_url": "https://github.com/test/repo.git", "repo_id": "test_repo"}))
    mock_walk.return_value = [("/tmp/repos/test_repo", [], ["test.py"])]
    
    with patch("builtins.open", MagicMock()) as mock_open:
        mock_open.return_value.read.return_value = "test content"
        # Make the worker run only once
        mock_redis_worker.brpop.side_effect = [(None, json.dumps({"repo_url": "https://github.com/test/repo.git", "repo_id": "test_repo"})), redis.exceptions.TimeoutError]
        try:
            ingestion_worker()
        except redis.exceptions.TimeoutError:
            pass
    
    mock_clone.assert_called_once_with("https://github.com/test/repo.git", "/tmp/repos/test_repo")
    mock_redis_worker.lpush.assert_called()

@patch("ingestion.ingest.FAISS")
@patch("ingestion.ingest.HuggingFaceEmbeddings")
def test_create_vector_store_from_chunks(mock_hf_embeddings, mock_faiss, mock_redis_client):
    _, mock_redis_ingest = mock_redis_client
    mock_redis_ingest.brpop.return_value = (None, json.dumps({"repo_id": "test_repo", "file_path": "test.py", "chunk_index": 0, "text": "test content"}))
    
    # Make the worker run only once
    mock_redis_ingest.brpop.side_effect = [(None, json.dumps({"repo_id": "test_repo", "file_path": "test.py", "chunk_index": 0, "text": "test content"})), redis.exceptions.TimeoutError]
    
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False
        try:
            create_vector_store_from_chunks("huggingface", "/tmp/test_db")
        except redis.exceptions.TimeoutError:
            pass
    
    mock_faiss.from_texts.assert_called_once_with(["test content"], mock_hf_embeddings.return_value)
    mock_faiss.from_texts.return_value.save_local.assert_called_with("/tmp/test_db")