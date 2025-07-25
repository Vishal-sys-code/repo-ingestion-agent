import os
import shutil
import click
from git import Repo
from langchain.text_splitter import CharacterTextSplitter
import redis
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, VertexAIEmbeddings

import time
import stat

def on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.
    
    Usage: shutil.rmtree(path, onerror=on_rm_error)
    """
    # Check if the error is due to a read-only file
    if not os.access(path, os.W_OK):
        # I can't change the access rights on this file, so we are stuck with it
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

def clone_repo(repo_url, local_path):
    """
    Clone a git repository to a local path.
    """
    if os.path.exists(local_path):
        shutil.rmtree(local_path, onerror=on_rm_error)
    Repo.clone_from(repo_url, local_path)

def load_and_chunk_documents(repo_path, repo_id):
    """
    Load documents from a repository and chunk them.
    """
    r = redis.Redis(host="127.0.0.1", port=6379, db=0)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    chunks = text_splitter.split_text(text)
                    for i, chunk in enumerate(chunks):
                        task = {
                            "repo_id": repo_id,
                            "file_path": os.path.join(root, file),
                            "chunk_index": i,
                            "text": chunk,
                        }
                        r.lpush("ingest:chunks", json.dumps(task))

@click.command()
@click.option("--repo_url", help="The URL of the git repository to ingest.")
@click.option("--repo_id", help="The ID of the repository.")
def main(repo_url, repo_id):
    """
    Main function to ingest a repository.
    """
    local_path = f"/tmp/{repo_id}"
    clone_repo(repo_url, local_path)
    load_and_chunk_documents(local_path, repo_id)

if __name__ == "__main__":
    main()