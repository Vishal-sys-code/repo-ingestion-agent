from celery import Celery
from ingestion.worker import ingestion_worker, enqueue_repo
import redis, json

celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")

@celery_app.task
def ingestion_worker_task():
    """
    Celery task to run the ingestion worker.
    """
    ingestion_worker()

@celery_app.task
def enqueue_repo_task(repo_url, repo_id):
    """
    Celery task to enqueue a repository for ingestion.
    """
    enqueue_repo(repo_url, repo_id)
