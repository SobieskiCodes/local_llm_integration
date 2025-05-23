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
import time
import hashlib

# ✅ ENHANCED Environment settings with optimized timeouts
OLLAMA_URL = os.getenv("OLLAMA_API", "http://ollama:11434")
OLLAMA_TIMEOUT = 180      # Increased from 60 for better reliability
EMBEDDING_TIMEOUT = 90    # Increased from 30 for better reliability

# ✅ PERFORMANCE OPTIMIZATIONS
CONTEXT_SIZE = 1024       # Reduced from 2048 for speed
MAX_TOKENS = 256         # Limit response length for speed
SIMILARITY_TOP_K = 2     # Reduced from 3 for speed
CACHE_EXPIRY = 300       # 5 minutes cache

print(f"🔧 Configuring OPTIMIZED Ollama connection...")
print(f"📊 Settings: LLM timeout={OLLAMA_TIMEOUT}s, Embedding timeout={EMBEDDING_TIMEOUT}s")
print(f"⚡ Performance: Context={CONTEXT_SIZE}, MaxTokens={MAX_TOKENS}, TopK={SIMILARITY_TOP_K}")

# ✅ OPTIMIZED Embedding model with performance settings
try:
    Settings.embed_model = OllamaEmbedding(
        model_name="mistral:latest", 
        base_url=OLLAMA_URL,
        request_timeout=EMBEDDING_TIMEOUT,
        # Performance optimizations
        num_ctx=CONTEXT_SIZE,
        repeat_penalty=1.1
    )
    print("✅ Optimized embedding model configured")
except Exception as e:
    print(f"❌ Embedding model configuration failed: {e}")

# ✅ OPTIMIZED LLM with keep-alive and performance settings
try:
    llm = Ollama(
        model="mistral:latest", 
        base_url=OLLAMA_URL,
        request_timeout=OLLAMA_TIMEOUT,
        temperature=0.1,
        # CRITICAL PERFORMANCE OPTIMIZATIONS
        num_ctx=CONTEXT_SIZE,        # Reduced context for speed
        num_predict=MAX_TOKENS,      # Limit response length  
        repeat_penalty=1.1,
        top_k=20,                    # Reduce sampling space
        top_p=0.9,                   # Focus on likely tokens
        keep_alive="24h"             # KEEP MODEL LOADED!
    )
    Settings.llm = llm
    print("✅ Optimized LLM configured with keep_alive=24h")
except Exception as e:
    print(f"❌ LLM configuration failed: {e}")

# ✅ ENHANCED Qdrant client setup
print("🔧 Setting up ENHANCED Qdrant connection...")
try:
    qdrant_client = QdrantClient(host="qdrant", port=6333)
    COLLECTION_NAME = "chat_memory"
    
    # ✅ Ensure collection exists with optimized settings
    existing_collections = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        print(f"🛠️ Creating OPTIMIZED collection '{COLLECTION_NAME}' in Qdrant...")
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=4096, 
                distance=Distance.COSINE,
                # Performance optimizations
                on_disk=True,  # Use disk storage for large datasets
            )
        )
    print("✅ Optimized Qdrant collection ready")
except Exception as e:
    print(f"❌ Qdrant setup failed: {e}")

# ✅ OPTIMIZED Storage & Vector setup
try:
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # ✅ Create optimized index from existing Qdrant data
    index = VectorStoreIndex.from_vector_store(vector_store)
    print("✅ Optimized vector index loaded")
except Exception as e:
    print(f"❌ Vector store setup failed: {e}")

# ✅ OPTIMIZED Chat engine with performance settings
try:
    memory = ChatMemoryBuffer.from_defaults(token_limit=800)  # Reduced for speed
    chat_engine = ContextChatEngine.from_defaults(
        llm=llm,
        memory=memory,
        retriever=index.as_retriever(
            similarity_top_k=SIMILARITY_TOP_K,  # Reduced for speed
            response_mode="compact"             # Faster response mode
        )
    )
    print("✅ Optimized chat engine initialized")
except Exception as e:
    print(f"❌ Chat engine setup failed: {e}")

app = FastAPI(
    title="Optimized LlamaIndex RAG Service",
    description="High-performance RAG with caching and optimization",
    version="2.1.0-optimized"
)

# ✅ SIMPLE QUERY CACHE for performance
query_cache = {}

def get_cache_key(query: str) -> str:
    """Generate cache key for query"""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()

def is_cache_valid(timestamp: float) -> bool:
    """Check if cache entry is still valid"""
    return time.time() - timestamp < CACHE_EXPIRY

# ✅ ENHANCED helper: Store new doc in Qdrant with error handling
def store_to_qdrant(query: str, response: str):
    """Store query-response pair in Qdrant with enhanced error handling"""
    try:
        doc = Document(
            text=f"Q: {query}\nA: {response}",
            metadata={
                "source": "chat", 
                "type": "qa_pair",
                "timestamp": time.time()
            }
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

# ✅ OPTIMIZED chat endpoint with caching and enhanced error handling
@app.post("/query")
async def optimized_query(request: Request):
    body = await request.json()
    query_text = body.get("input")
    
    if not query_text:
        return {"result": "No input provided"}
    
    print(f"📥 Received query: {query_text}")
    
    # ✅ CHECK CACHE FIRST
    cache_key = get_cache_key(query_text)
    if cache_key in query_cache:
        cached_response, timestamp = query_cache[cache_key]
        if is_cache_valid(timestamp):
            print(f"🚀 Cache hit for: {query_text[:30]}...")
            return {"result": cached_response}
        else:
            # Remove expired cache entry
            del query_cache[cache_key]

    start_time = time.time()
    
    try:
        print("🤖 Processing with optimized chat engine...")
        
        # Enhanced timeout handling around the chat call
        response = chat_engine.chat(query_text)
        response_text = str(response)
        
        duration = time.time() - start_time
        print(f"✅ Chat response generated in {duration:.2f}s: {len(response_text)} characters")
        
        # ✅ CACHE THE RESULT
        query_cache[cache_key] = (response_text, time.time())
        
        # ✅ CLEAN OLD CACHE ENTRIES (simple cleanup)
        if len(query_cache) > 100:
            # Remove oldest entry
            oldest_key = min(query_cache.keys(), key=lambda k: query_cache[k][1])
            del query_cache[oldest_key]
        
        # Store the successful interaction
        store_to_qdrant(query_text, response_text)
        
        return {"result": response_text}
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Chat engine failed after {duration:.2f}s: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Enhanced fallback response
        fallback_response = f"I apologize, but I encountered a technical issue processing your query: '{query_text}'. This might be due to high load or a complex query. Please try again with a simpler question. Error: {str(e)[:100]}..."
        
        return {"result": fallback_response}

# ✅ ENHANCED health check endpoint with comprehensive status
@app.get("/health")
async def enhanced_health():
    health_status = {
        "status": "ok",
        "version": "2.1.0-optimized",
        "llm_model": Settings.llm.model,
        "vector_store": COLLECTION_NAME,
        "optimizations": {
            "ollama_timeout": f"{OLLAMA_TIMEOUT}s",
            "embedding_timeout": f"{EMBEDDING_TIMEOUT}s",
            "context_size": CONTEXT_SIZE,
            "max_tokens": MAX_TOKENS,
            "similarity_top_k": SIMILARITY_TOP_K,
            "keep_alive": "24h",
            "cache_enabled": True,
            "cache_expiry": f"{CACHE_EXPIRY}s"
        },
        "cache_stats": {
            "entries": len(query_cache),
            "max_entries": 100
        }
    }
    
    # Test Ollama connectivity
    try:
        import requests
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        health_status["ollama_connectivity"] = "ok" if response.status_code == 200 else "error"
        
        # Get model status
        ps_response = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        if ps_response.status_code == 200:
            models = ps_response.json().get("models", [])
            health_status["ollama_models_loaded"] = len(models)
            health_status["mistral_loaded"] = any("mistral" in m.get("name", "") for m in models)
        
    except Exception as e:
        health_status["ollama_connectivity"] = f"error: {str(e)}"
        health_status["mistral_loaded"] = False
    
    # Test Qdrant connectivity
    try:
        collections = qdrant_client.get_collections()
        health_status["qdrant_connectivity"] = "ok"
        health_status["qdrant_collections"] = len(collections.collections)
        
        # Get collection info
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        health_status["vector_count"] = collection_info.vectors_count
        
    except Exception as e:
        health_status["qdrant_connectivity"] = f"error: {str(e)}"
        health_status["vector_count"] = "unknown"
    
    return health_status

# ✅ ENHANCED configuration endpoint
@app.get("/config")
async def show_optimized_config():
    """Show current optimized configuration"""
    return {
        "ollama": {
            "url": OLLAMA_URL,
            "timeout": OLLAMA_TIMEOUT,
            "model": "mistral:latest",
            "context_size": CONTEXT_SIZE,
            "max_tokens": MAX_TOKENS,
            "keep_alive": "24h"
        },
        "qdrant": {
            "collection": COLLECTION_NAME,
            "similarity_top_k": SIMILARITY_TOP_K,
            "vector_size": 4096
        },
        "performance": {
            "cache_enabled": True,
            "cache_expiry": f"{CACHE_EXPIRY}s",
            "response_mode": "compact",
            "memory_token_limit": 800
        },
        "optimizations_applied": [
            "reduced_context_size",
            "limited_response_tokens", 
            "query_result_caching",
            "model_keep_alive",
            "compact_response_mode",
            "reduced_similarity_results"
        ]
    }

# ✅ ENHANCED test endpoint for debugging with performance metrics
@app.post("/test")
async def enhanced_test_endpoint(request: Request):
    """Enhanced test endpoint with performance metrics"""
    body = await request.json()
    query_text = body.get("input", "test query")
    
    test_results = {
        "query": query_text,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "optimizations": "enabled"
    }
    
    # Test 1: Direct LLM call with timing
    try:
        print("🧪 Testing optimized direct LLM call...")
        start_time = time.time()
        simple_response = llm.complete(query_text)
        duration = time.time() - start_time
        
        test_results["llm_direct"] = {
            "status": "success",
            "duration_seconds": round(duration, 2),
            "response_length": len(str(simple_response)),
            "response_preview": str(simple_response)[:100] + "..."
        }
    except Exception as e:
        test_results["llm_direct"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # Test 2: Chat engine call with timing
    try:
        print("🧪 Testing optimized chat engine...")
        start_time = time.time()
        chat_response = chat_engine.chat(query_text)
        duration = time.time() - start_time
        
        test_results["chat_engine"] = {
            "status": "success", 
            "duration_seconds": round(duration, 2),
            "response_length": len(str(chat_response)),
            "response_preview": str(chat_response)[:100] + "..."
        }
    except Exception as e:
        test_results["chat_engine"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # Test 3: Cache test
    cache_key = get_cache_key(query_text)
    test_results["cache_test"] = {
        "cache_key": cache_key,
        "in_cache": cache_key in query_cache,
        "cache_size": len(query_cache)
    }
    
    return test_results

# ✅ CACHE management endpoint
@app.post("/cache/clear")
async def clear_cache():
    """Clear the query cache"""
    global query_cache
    cache_size = len(query_cache)
    query_cache = {}
    return {
        "message": f"Cache cleared - removed {cache_size} entries",
        "cache_size": 0
    }

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "cache_entries": len(query_cache),
        "max_entries": 100,
        "expiry_seconds": CACHE_EXPIRY,
        "cache_keys": list(query_cache.keys())[:10]  # Show first 10 keys
    }

print("🚀 OPTIMIZED LlamaIndex service startup complete!")
print(f"📊 Performance settings: Context={CONTEXT_SIZE}, Tokens={MAX_TOKENS}, TopK={SIMILARITY_TOP_K}")
print(f"🚀 Cache enabled: {CACHE_EXPIRY}s expiry, 100 max entries")
print(f"⚡ Keep-alive enabled: Model stays loaded for 24h")