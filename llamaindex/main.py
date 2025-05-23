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
import logging

# ✅ Environment settings
OLLAMA_URL = os.getenv("OLLAMA_API", "http://ollama:11434")

# ✅ Configure longer timeouts for Ollama
OLLAMA_TIMEOUT = 60  # Increased timeout
EMBEDDING_TIMEOUT = 30

print(f"🔧 Configuring Ollama connection with {OLLAMA_TIMEOUT}s timeout...")

# ✅ LLM + Embedding config with longer timeouts
try:
    Settings.embed_model = OllamaEmbedding(
        model_name="mistral:latest", 
        base_url=OLLAMA_URL,
        request_timeout=EMBEDDING_TIMEOUT
    )
    print("✅ Embedding model configured")
except Exception as e:
    print(f"❌ Embedding model configuration failed: {e}")

try:
    llm = Ollama(
        model="mistral:latest", 
        base_url=OLLAMA_URL,
        request_timeout=OLLAMA_TIMEOUT,  # Key fix: longer timeout
        temperature=0.1,
        num_ctx=2048  # Limit context to speed up responses
    )
    Settings.llm = llm
    print("✅ LLM configured with extended timeout")
except Exception as e:
    print(f"❌ LLM configuration failed: {e}")

# ✅ Qdrant client setup
print("🔧 Setting up Qdrant connection...")
try:
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
    print("✅ Qdrant collection ready")
except Exception as e:
    print(f"❌ Qdrant setup failed: {e}")

# ✅ Storage & Vector setup
try:
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # ✅ Create index from existing Qdrant data
    index = VectorStoreIndex.from_vector_store(vector_store)
    print("✅ Vector index loaded")
except Exception as e:
    print(f"❌ Vector store setup failed: {e}")

# ✅ Chat engine with memory and timeout handling
try:
    memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
    chat_engine = ContextChatEngine.from_defaults(
        llm=llm,
        memory=memory,
        retriever=index.as_retriever(similarity_top_k=3)  # Limit results for speed
    )
    print("✅ Chat engine initialized")
except Exception as e:
    print(f"❌ Chat engine setup failed: {e}")

app = FastAPI()

# ✅ Helper: Store new doc in Qdrant
def store_to_qdrant(query: str, response: str):
    """Store query-response pair in Qdrant with error handling"""
    try:
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
        print(f"💾 Stored Q&A pair in vector store")
    except Exception as e:
        print(f"⚠️ Failed to store in Qdrant: {e}")

# ✅ Enhanced chat endpoint with timeout handling
@app.post("/query")
async def query(request: Request):
    body = await request.json()
    query_text = body.get("input")
    print(f"📥 Received query: {query_text}")

    try:
        # Add timeout handling around the chat call
        print("🤖 Querying chat engine...")
        
        response = chat_engine.chat(query_text)
        response_text = str(response)
        
        print(f"✅ Chat response generated: {len(response_text)} characters")
        
        # Store the successful interaction
        store_to_qdrant(query_text, response_text)
        
        return {"result": response_text}
        
    except Exception as e:
        error_msg = f"Chat engine failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Return a helpful error message instead of crashing
        fallback_response = f"I apologize, but I'm experiencing technical difficulties processing your query: '{query_text}'. The error was: {str(e)[:100]}..."
        
        return {"result": fallback_response}

# ✅ Enhanced health check endpoint
@app.get("/health")
async def health():
    health_status = {
        "status": "ok",
        "llm": Settings.llm.model,
        "vector_store": COLLECTION_NAME,
        "timeouts": {
            "ollama_llm": f"{OLLAMA_TIMEOUT}s",
            "embedding": f"{EMBEDDING_TIMEOUT}s"
        }
    }
    
    # Test Ollama connectivity
    try:
        # Quick test to see if Ollama is responding
        import requests
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        health_status["ollama_connectivity"] = "ok" if response.status_code == 200 else "error"
    except Exception as e:
        health_status["ollama_connectivity"] = f"error: {str(e)}"
    
    # Test Qdrant connectivity
    try:
        collections = qdrant_client.get_collections()
        health_status["qdrant_connectivity"] = "ok"
        health_status["qdrant_collections"] = len(collections.collections)
    except Exception as e:
        health_status["qdrant_connectivity"] = f"error: {str(e)}"
    
    return health_status

# ✅ Test endpoint for debugging
@app.post("/test")
async def test_endpoint(request: Request):
    """Test endpoint for debugging connectivity issues"""
    body = await request.json()
    query_text = body.get("input", "test query")
    
    test_results = {
        "query": query_text,
        "timestamp": "2025-05-22T19:30:00Z"
    }
    
    # Test 1: Simple LLM call
    try:
        print("🧪 Testing direct LLM call...")
        simple_response = llm.complete(query_text)
        test_results["llm_direct"] = {
            "status": "success",
            "response_length": len(str(simple_response)),
            "response_preview": str(simple_response)[:100] + "..."
        }
    except Exception as e:
        test_results["llm_direct"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # Test 2: Chat engine call
    try:
        print("🧪 Testing chat engine...")
        chat_response = chat_engine.chat(query_text)
        test_results["chat_engine"] = {
            "status": "success", 
            "response_length": len(str(chat_response)),
            "response_preview": str(chat_response)[:100] + "..."
        }
    except Exception as e:
        test_results["chat_engine"] = {
            "status": "failed",
            "error": str(e)
        }
    
    return test_results

print("🚀 LlamaIndex service startup complete!")
print(f"📊 Configuration: Ollama timeout={OLLAMA_TIMEOUT}s, Embedding timeout={EMBEDDING_TIMEOUT}s")