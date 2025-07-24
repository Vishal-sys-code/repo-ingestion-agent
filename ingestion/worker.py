import os, json, redis
from git import Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Redis queue
r = redis.Redis(host="localhost", port=6379, db=0)

def enqueue_repo(repo_url, repo_id):
    payload = {"repo_url": repo_url, "repo_id": repo_id}
    r.lpush("ingest:repos", json.dumps(payload))

def ingestion_worker():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    while True:
        _, raw = r.brpop("ingest:repos")
        task = json.loads(raw)
        path = f"/tmp/repos/{task['repo_id']}"
        # 1️⃣ Clone or pull
        if os.path.isdir(path):
            Repo(path).remotes.origin.pull()
        else:
            Repo.clone_from(task["repo_url"], path)
        # 2️⃣ Chunk files
        for dirpath, _, files in os.walk(path):
            for fn in files:
                if fn.endswith((".py", ".js", ".ts")):
                    full = os.path.join(dirpath, fn)
                    text = open(full, encoding="utf‑8").read()
                    chunks = splitter.split_text(text)
                    for i, chunk in enumerate(chunks):
                        msg = {
                          "repo_id": task["repo_id"],
                          "file_path": os.path.relpath(full, path),
                          "chunk_index": i,
                          "text": chunk
                        }
                        r.lpush("ingest:chunks", json.dumps(msg))
        print(f"[🔍] {task['repo_id']} ingestion enqueued {len(chunks)} chunks")
