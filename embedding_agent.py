import redis, json
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import VertexAIEmbeddings

# A) HuggingFace from langchain.embeddings import HuggingFaceInstructEmbeddings
emb_model_huggingface = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-xl",
    model_kwargs={"device": "cpu"}
)

# B) Google Gemini (via Vertex AI) from langchain.embeddings import VertexAIEmbeddings
emb_model_gemini = VertexAIEmbeddings(
    # project="YOUR_GCP_PROJECT",  # Will be set from config
    location="us-central1",
    model_name="gemini-pro-embeddings-beta"
)

r = redis.Redis(host="localhost", port=6379, db=0)

# On-disk index + SQLite for metadata
vectordb = FAISS(
    embedding_function=emb_model_huggingface,  # Default to HuggingFace
    index_path="faiss.index",
    docstore_path="faiss_meta.db"
)

def embedding_worker():
    while True:
        _, raw = r.brpop("ingest:chunks")
        task = json.loads(raw)
        doc = Document(
            page_content=task["text"],
            metadata={
                "repo_id": task["repo_id"],
                "file_path": task["file_path"],
                "chunk_index": task["chunk_index"]
            }
        )
        vectordb.add_documents([doc], ids=[f"{task['repo_id']}/{task['file_path']}/{task['chunk_index']}"])
