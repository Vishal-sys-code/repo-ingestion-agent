from celery import Celery
from ingestion.ingest import main as ingest_repo

celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")

@celery_app.task
def ingest_repo_task(repo_url, embeddings_type, db_path):
    """
    Celery task to ingest a repository.
    """
    ingest_repo.main(
        [
            "--repo_url",
            repo_url,
            "--embeddings_type",
            embeddings_type,
            "--db_path",
            db_path,
        ],
        standalone_mode=False,
    )
