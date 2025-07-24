import os
import shutil
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.ingest import clone_repo, load_documents, create_vector_store, on_rm_error

def test_clone_repo():
    repo_url = "https://github.com/langchain-ai/langchain.git"
    local_path = "/tmp/test_repo"
    clone_repo(repo_url, local_path)
    assert os.path.exists(local_path)
    shutil.rmtree(local_path, onerror=on_rm_error)

def test_load_documents():
    repo_path = "/tmp/test_load_documents"
    os.makedirs(repo_path, exist_ok=True)
    with open(os.path.join(repo_path, "test.py"), "w") as f:
        f.write("test content")
    documents = load_documents(repo_path)
    assert len(documents) == 1
    assert documents[0] == "test content"
    shutil.rmtree(repo_path)

@patch("ingestion.ingest.FAISS.from_documents")
@patch("ingestion.ingest.HuggingFaceEmbeddings")
@patch("ingestion.ingest.VertexAIEmbeddings")
def test_create_vector_store(mock_vertex_embeddings, mock_hf_embeddings, mock_faiss):
    documents = ["test content"]
    db_path = "/tmp/test_db"
    
    mock_hf_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock_vertex_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    # Test HuggingFace embeddings
    create_vector_store(documents, "huggingface", db_path)
    mock_hf_embeddings.assert_called_once()
    mock_faiss.return_value.save_local.assert_called_with(db_path)

    # Test Google embeddings
    create_vector_store(documents, "google", db_path)
    mock_vertex_embeddings.assert_called_once()
    mock_faiss.return_value.save_local.assert_called_with(db_path)