from fastapi import FastAPI
from pydantic import BaseModel
from ingestion.tasks import ingest_repo_task

app = FastAPI()

class Repo(BaseModel):
    repo_url: str
    embeddings_type: str = "huggingface"
    db_path: str = "faiss_index"

@app.post("/ingest")
async def ingest(repo: Repo):
    ingest_repo_task.delay(repo.repo_url, repo.embeddings_type, repo.db_path)
    return {"message": "Repository ingestion started."}
