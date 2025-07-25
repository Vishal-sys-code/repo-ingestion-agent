from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings,GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

# Load the vector database
vectordb = FAISS.load_local("faiss.index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)

# Build retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Define QA prompt
template = """You are a code-savvy AI. Given these code snippets from {repo_id} and the question below, answer concisely with references:

{context}

Question: {question}
Answer:"""
prompt = PromptTemplate(
    input_variables=["repo_id", "context", "question"],
    template=template
)

qa = RetrievalQA.from_chain_type(
    llm=GoogleGenerativeAI(model="gemini-pro"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context"
    }
)

def retrieve(repo_id, question):
    result = qa.invoke({"repo_id": repo_id, "query": question})
    # The result is a dictionary containing the answer and the source documents.
    # The answer is in the "result" key.
    print(result["result"])
    # Optionally show source docs
    for doc in result["source_documents"]:
        print(f"— {doc.metadata['file_path']}#{doc.metadata['chunk_index']}")

if __name__ == '__main__':
    # This is an example of how to run the retrieval agent.
    # You will need to have a FAISS index built first.
    # You can build one by running the ingestion script:
    # python -m ingestion.ingest --repo_url https://github.com/langchain-ai/langchain --repo_id langchain

    print("--- Testing retrieval agent ---")
    retrieve(repo_id="langchain", question="How does the text splitting work?")
