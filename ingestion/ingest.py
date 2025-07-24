import os
import shutil
import click
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, VertexAIEmbeddings
import redis
import json
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

def create_vector_store_from_chunks(embeddings_type, db_path):
    """
    Create a FAISS vector store from chunks in a Redis queue.
    """
    r = redis.Redis(host="localhost", port=6379, db=0)
    
    if embeddings_type == "huggingface":
        embeddings = HuggingFaceEmbeddings()
    elif embeddings_type == "google":
        embeddings = VertexAIEmbeddings()
    else:
        raise ValueError("Invalid embeddings type")

    while True:
        _, raw = r.brpop("ingest:chunks")
        chunk = json.loads(raw)
        
        db = FAISS.from_texts([chunk['text']], embeddings)
        if os.path.exists(db_path):
            local_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            local_db.merge_from(db)
            local_db.save_local(db_path)
        else:
            db.save_local(db_path)

@click.command()
@click.option("--embeddings_type", default="huggingface", help="The type of embeddings to use (huggingface or google).")
@click.option("--db_path", default="faiss_index", help="The path to the FAISS vector store.")
def main(embeddings_type, db_path):
    """
    Main function to create a vector store from chunks.
    """
    create_vector_store_from_chunks(embeddings_type, db_path)

if __name__ == "__main__":
    main()