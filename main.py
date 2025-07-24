from fastapi import FastAPI
from pydantic import BaseModel
from ingestion.tasks import enqueue_repo_task
import uuid

app = FastAPI()

class Repo(BaseModel):
    repo_url: str

@app.post("/ingest")
async def ingest(repo: Repo):
    repo_id = str(uuid.uuid4())
    enqueue_repo_task.delay(repo.repo_url, repo_id)
    return {"message": "Repository ingestion started.", "repo_id": repo_id}