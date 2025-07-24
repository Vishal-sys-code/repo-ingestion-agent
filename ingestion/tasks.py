from celery import Celery
from ingestion.ingest import main as ingest_repo
from embedding_agent import embedding_worker

celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")

@celery_app.task
def ingest_repo_task(repo_url, repo_id):
    """
    Celery task to ingest a repository.
    """
    ingest_repo.main(
        [
            "--repo_url",
            repo_url,
            "--repo_id",
            repo_id,
        ],
        standalone_mode=False,
    )

@celery_app.task
def embedding_worker_task():
    """
    Celery task to run the embedding worker.
    """
    embedding_worker()
