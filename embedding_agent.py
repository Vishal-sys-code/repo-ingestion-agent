import redis, json, os
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import VertexAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

# Google Gemini (via Vertex AI) from langchain.embeddings import VertexAIEmbeddings
emb_model_gemini = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

import faiss

# On-disk index + SQLite for metadata
if os.path.exists("faiss.index"):
    vectordb = FAISS.load_local("faiss.index", emb_model_gemini, allow_dangerous_deserialization=True)
else:
    index = faiss.IndexFlatL2(768)  # Assuming the embedding dimension is 768
    vectordb = FAISS(embedding_function=emb_model_gemini, index=index, docstore={}, index_to_docstore_id={})
    vectordb.save_local("faiss.index")


def embedding_worker(r):
    while True:
        try:
            _, raw = r.brpop("ingest:chunks")
            task = json.loads(raw)
        except StopIteration:
            break
        doc = Document(
            page_content=task["text"],
            metadata={
                "repo_id": task["repo_id"],
                "file_path": task["file_path"],
                "chunk_index": task["chunk_index"]
            }
        )
        vectordb.add_documents([doc], ids=[f"{task['repo_id']}/{task['file_path']}/{task['chunk_index']}"])
        vectordb.save_local("faiss.index")

if __name__ == "__main__":
    r = redis.Redis(host="127.0.0.1", port=6379, db=0)
    embedding_worker(r)
