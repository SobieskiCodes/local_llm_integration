from fastapi import FastAPI, Request
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
import os

# ✅ Environment settings
OLLAMA_URL = os.getenv("OLLAMA_API", "http://ollama:11434")

# ✅ LLM + Embedding config
Settings.embed_model = OllamaEmbedding(model_name="mistral:latest", base_url=OLLAMA_URL)
llm = Ollama(model="mistral:latest", base_url=OLLAMA_URL)
Settings.llm = llm

# ✅ Qdrant client setup
qdrant_client = QdrantClient(host="qdrant", port=6333)
COLLECTION_NAME = "chat_memory"

# ✅ Ensure collection exists
existing_collections = [c.name for c in qdrant_client.get_collections().collections]
if COLLECTION_NAME not in existing_collections:
    print(f"🛠️ Creating collection '{COLLECTION_NAME}' in Qdrant...")
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=4096, distance=Distance.COSINE)
    )

# ✅ Storage & Vector setup
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ✅ Create index from existing Qdrant data
index = VectorStoreIndex.from_vector_store(vector_store)

# ✅ Chat engine with memory
memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
chat_engine = ContextChatEngine.from_defaults(
    llm=llm,
    memory=memory,
    retriever=index.as_retriever()
)

app = FastAPI()

# ✅ Helper: Store new doc in Qdrant
def store_to_qdrant(query: str, response: str):
    doc = Document(
        text=f"Q: {query}\nA: {response}",
        metadata={"source": "chat", "type": "qa_pair"}
    )
    # Append this single doc to Qdrant using existing storage context
    VectorStoreIndex.from_documents(
        [doc],
        storage_context=storage_context,
        vector_store=vector_store,
        show_progress=False
    )

# ✅ Chat endpoint
@app.post("/query")
async def query(request: Request):
    body = await request.json()
    query_text = body.get("input")
    print("📥 Received query from LangChain:", query_text)

    response = chat_engine.chat(query_text)
    response_text = str(response)

    store_to_qdrant(query_text, response_text)

    return {"result": response_text}

# ✅ Health check endpoint
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm": Settings.llm.model,
        "vector_store": COLLECTION_NAME
    }
