import os
import shutil
import click
from git import Repo
from langchain.text_splitter import CharacterTextSplitter
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

def load_documents(repo_path):
    """
    Load documents from a repository.
    """
    documents = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                    documents.append(f.read())
    return documents

def create_vector_store(documents, embeddings_type, db_path):
    """
    Create a FAISS vector store.
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)
    
    if embeddings_type == "huggingface":
        embeddings = HuggingFaceEmbeddings()
    elif embeddings_type == "google":
        embeddings = VertexAIEmbeddings()
    else:
        raise ValueError("Invalid embeddings type")

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(db_path)

@click.command()
@click.option("--repo_url", help="The URL of the git repository to ingest.")
@click.option("--embeddings_type", default="huggingface", help="The type of embeddings to use (huggingface or google).")
@click.option("--db_path", default="faiss_index", help="The path to the FAISS vector store.")
def main(repo_url, embeddings_type, db_path):
    """
    Main function to ingest a repository.
    """
    local_path = "/tmp/repo"
    clone_repo(repo_url, local_path)
    documents = load_documents(local_path)
    create_vector_store(documents, embeddings_type, db_path)

if __name__ == "__main__":
    main()