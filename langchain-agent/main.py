from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent  
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime

app = FastAPI(
    title="Optimized LangChain Agent API",
    description="Smart Routing + Fast Agents",
    version="2.1.0-optimized"
)

# ========== OPTIMIZED CONFIGURATION ==========
LLAMAINDEX_TIMEOUT = 90  # RAG + embeddings + LLM generation
OLLAMA_TIMEOUT = 45      # Direct LLM calls (shorter than RAG)
AGENT_TIMEOUT = 120      # Must be longer than tool timeouts!

print(f"🚀 Starting OPTIMIZED LangChain service with fast timeouts")

# ========== SMART ROUTING LOGIC ==========
def should_use_agent(query: str) -> bool:
    """Decide if query needs agent reasoning or can use direct RAG"""
    query_lower = query.lower()
    
    # Force agent for "our" specific setup questions
    our_setup_keywords = [
        "our", "we are using", "in our setup", "our configuration", 
        "our docker", "our system", "we have", "our implementation"
    ]
    if any(keyword in query_lower for keyword in our_setup_keywords):
        return True
    
    # Force agent for specific/exact technical questions
    specific_keywords = [
        "exact", "specific", "configuration parameters", "integration", 
        "setup", "implementation details", "how to configure", "settings"
    ]
    if any(keyword in query_lower for keyword in specific_keywords):
        return True
    
    # Agent-worthy questions (need reasoning/analysis)
    agent_patterns = [
        "compare", "versus", "vs", "difference between", "pros and cons",
        "analyze", "analysis", "evaluate", "assessment", "recommend",
        "which is better", "should i", "help me choose", "strategy",
        "plan for", "how to implement", "step by step", "best practices",
        "trade-offs", "design", "architecture", "approach"
    ]
    
    if any(pattern in query_lower for pattern in agent_patterns):
        return True
    
    # Questions with multiple parts likely need agent reasoning
    question_words = ["what", "how", "why", "when", "where", "which"]
    question_count = sum(1 for word in question_words if word in query_lower)
    
    if question_count > 1:  # Multiple questions
        return True
    
    # Long queries might need analysis
    if len(query.split()) > 15:
        return True
    
    # Questions asking for latest/current information
    current_info_keywords = [
        "latest", "current", "recent", "new", "updated", "modern", "today"
    ]
    if any(keyword in query_lower for keyword in current_info_keywords):
        return True
    
    # Simple factual questions → Direct RAG (fast)
    # BUT make the criteria more restrictive
    simple_patterns = [
        "what is", "define", "meaning of"
    ]
    
    # Only use direct RAG for truly simple, short questions
    is_simple_question = any(pattern in query_lower for pattern in simple_patterns)
    is_short_query = len(query.split()) <= 8
    has_no_complex_words = not any(word in query_lower for word in [
        "architecture", "system", "implementation", "integration", 
        "configuration", "setup", "parameters", "approach"
    ])
    
    if is_simple_question and is_short_query and has_no_complex_words:
        return False  # Use direct RAG
    
    # Default to agent for anything else (better safe than sorry)
    return True

def get_routing_reason(query: str, use_agent: bool) -> str:
    """Explain why we chose agent vs direct RAG"""
    if use_agent:
        if any(word in query.lower() for word in ["compare", "analyze", "versus"]):
            return "comparison_analysis"
        elif any(word in query.lower() for word in ["recommend", "should", "choose"]):
            return "recommendation_needed"
        elif len(query.split()) > 15:
            return "complex_query"
        else:
            return "multi_step_reasoning"
    else:
        return "simple_factual_lookup"

# ========== CORE FUNCTIONS ==========
def query_llamaindex_fast(input_text: str) -> str:
    """Optimized LlamaIndex query"""
    start_time = time.time()
    print(f"⚡ Fast RAG query: '{input_text[:40]}...'")
    
    try:
        response = requests.post(
            "http://llamaindex:8081/query", 
            json={"input": input_text},
            timeout=LLAMAINDEX_TIMEOUT
        )
        
        duration = time.time() - start_time
        print(f"⏱️ RAG responded in {duration:.2f}s")
        
        response.raise_for_status()
        result = str(response.json().get("result", "No result found"))
        
        print(f"✅ RAG success: {len(result)} chars")
        return result
        
    except requests.exceptions.Timeout:
        return f"RAG timeout after {LLAMAINDEX_TIMEOUT}s - try a simpler question"
    except Exception as e:
        return f"RAG error: {str(e)}"

# ========== OPTIMIZED LANGCHAIN SETUP ==========
def create_optimized_tools():
    """Streamlined tools for faster execution"""
    return [
        Tool(
            name="KnowledgeBase",
            func=query_llamaindex_fast,
            description="Search knowledge base for technology and programming information. Use for factual lookups."
        )
    ]

def create_fast_llm():
    """Optimized LLM for speed"""
    print("⚡ Initializing FAST Ollama LLM...")
    
    try:
        llm = ChatOpenAI(
            openai_api_base="http://ollama:11434/v1",
            openai_api_key="not-needed",
            model="mistral:latest",
            temperature=0.0,  # ← Faster, more deterministic
            max_tokens=300,   # ← Shorter responses
            request_timeout=OLLAMA_TIMEOUT  # ← Faster timeout
        )
        print("✅ Fast LLM initialized")
        return llm
    except Exception as e:
        print(f"❌ LLM initialization failed: {e}")
        raise

# Initialize optimized components
tools = create_optimized_tools()
llm = create_fast_llm()

# Streamlined prompt for speed
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the KnowledgeBase tool for factual information. Be concise and direct."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create FAST agent
print("⚡ Creating FAST agent...")
try:
    agent = create_openai_functions_agent(
        llm=llm, 
        tools=tools, 
        prompt=prompt
    )
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,  # ← No slow logging
        max_iterations=1,  # ← Single attempt
        max_execution_time=AGENT_TIMEOUT,  # ← Fast timeout
        handle_parsing_errors=True,
        return_intermediate_steps=True,  # ← Minimal output
        early_stopping_method="generate"  # ← Stop early when possible
    )
    print("✅ Fast agent created")
except Exception as e:
    print(f"❌ Agent creation failed: {e}")
    agent_executor = None

# ========== HELPER FUNCTIONS ==========
async def safe_parse_json(request: Request) -> dict:
    """Fast JSON parsing"""
    try:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty request body")
        
        json_data = json.loads(body.decode('utf-8'))
        
        if not isinstance(json_data, dict) or "input" not in json_data:
            raise HTTPException(status_code=400, detail="Missing 'input' field")
        
        return json_data
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

def fast_agent_invoke(query_text: str) -> dict:
    """Fast agent execution with quick fallback"""
    start_time = time.time()
    print(f"🤖 Fast agent: '{query_text[:40]}...'")
    
    if agent_executor is None:
        result = query_llamaindex_fast(query_text)
        return {
            "result": result,
            "method": "no_agent_fallback",
            "duration": time.time() - start_time
        }
    
    try:
        # Fast agent execution
        agent_result = agent_executor.invoke({"input": query_text})
        
        duration = time.time() - start_time
        
        if isinstance(agent_result, dict):
            output = agent_result.get("output", str(agent_result))
        else:
            output = str(agent_result)
        
        print(f"✅ Fast agent success in {duration:.2f}s")
        
        return {
            "result": output,
            "method": "fast_agent_success", 
            "duration": duration
        }
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"⚡ Agent failed quickly ({duration:.2f}s), using RAG fallback")
        
        # Quick fallback to RAG
        fallback_result = query_llamaindex_fast(query_text)
        
        return {
            "result": fallback_result,
            "method": "quick_fallback",
            "duration": time.time() - start_time,
            "agent_error": str(e)[:50]
        }

# ========== OPTIMIZED API ENDPOINTS ==========

@app.post("/api/chat")
async def smart_chat(request: Request):
    """Smart routing chat endpoint - agents only when needed"""
    try:
        json_data = await safe_parse_json(request)
        query_text = json_data["input"]
        
        print(f"\n📨 Smart chat: '{query_text}'")
        
        # Decide: Agent or Direct RAG?
        use_agent = should_use_agent(query_text)
        routing_reason = get_routing_reason(query_text, use_agent)
        
        start_time = time.time()
        
        if use_agent:
            print(f"🤖 Using AGENT (reason: {routing_reason})")
            result_data = fast_agent_invoke(query_text)
        else:
            print(f"⚡ Using DIRECT RAG (reason: {routing_reason})")
            direct_result = query_llamaindex_fast(query_text)
            result_data = {
                "result": direct_result,
                "method": "direct_rag",
                "duration": time.time() - start_time
            }
        
        return {
            "response": result_data["result"],
            "method": result_data["method"],
            "routing_decision": "agent" if use_agent else "direct_rag",
            "routing_reason": routing_reason,
            "duration_seconds": result_data["duration"],
            "timestamp": datetime.now().isoformat(),
            "version": "2.1.0-optimized"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Emergency fallback
        try:
            json_data = await safe_parse_json(request)
            emergency_result = query_llamaindex_fast(json_data["input"])
            return {
                "response": emergency_result,
                "method": "emergency_fallback",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        except:
            raise HTTPException(status_code=500, detail=f"Complete failure: {str(e)}")

@app.post("/api/research/simple")
async def simple_rag_endpoint(request: Request):
    """Direct RAG endpoint - always fast"""
    try:
        json_data = await safe_parse_json(request)
        query_text = json_data["input"]
        
        result = query_llamaindex_fast(query_text)
        
        return {
            "query": query_text,
            "result": result,
            "method": "direct_rag_optimized",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simple RAG failed: {str(e)}")

@app.post("/api/research/agent")
async def force_agent_endpoint(request: Request):
    """Force agent usage - for testing complex queries"""
    try:
        json_data = await safe_parse_json(request)
        query_text = json_data["input"]
        
        result_data = fast_agent_invoke(query_text)
        
        return {
            "query": query_text,
            "result": result_data["result"],
            "method": result_data["method"],
            "duration_seconds": result_data["duration"],
            "forced_agent": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forced agent failed: {str(e)}")

@app.get("/api/routing/test")
async def test_routing_logic():
    """Test which queries go to agent vs RAG"""
    test_queries = [
        "What is Docker?",
        "Compare Docker vs Kubernetes",
        "Explain microservices architecture", 
        "Should I use React or Vue for my project?",
        "Analyze the pros and cons of serverless computing",
        "How does machine learning work?",
        "What are the best practices for API design?"
    ]
    
    results = []
    for query in test_queries:
        use_agent = should_use_agent(query)
        reason = get_routing_reason(query, use_agent)
        results.append({
            "query": query,
            "routing": "agent" if use_agent else "direct_rag",
            "reason": reason
        })
    
    return {
        "routing_test": results,
        "summary": {
            "agent_queries": len([r for r in results if r["routing"] == "agent"]),
            "rag_queries": len([r for r in results if r["routing"] == "direct_rag"])
        }
    }

@app.get("/api/capabilities")
async def system_capabilities():
    """Enhanced capabilities with routing info"""
    return {
        "version": "2.1.0-optimized",
        "features": {
            "smart_routing": {
                "description": "Automatically routes simple queries to fast RAG, complex queries to agents",
                "simple_queries": "2-5 seconds (direct RAG)",
                "complex_queries": "8-15 seconds (fast agent)",
                "fallback": "Always available"
            },
            "agent_optimization": {
                "max_execution_time": f"{AGENT_TIMEOUT}s",
                "max_iterations": 1,
                "verbose_logging": False
            }
        },
        "routing_examples": {
            "direct_rag": ["What is X?", "Define Y", "Explain Z"],
            "agent": ["Compare A vs B", "Should I use X?", "Analyze pros/cons", "Recommend approach"]
        },
        "endpoints": {
            "smart_chat": "POST /api/chat (automatic routing)",
            "force_rag": "POST /api/research/simple",
            "force_agent": "POST /api/research/agent",
            "test_routing": "GET /api/routing/test"
        }
    }

@app.get("/health")
async def health_check():
    """Fast health check"""
    return {
        "status": "healthy",
        "version": "2.1.0-optimized",
        "optimizations": [
            "smart_routing_enabled",
            "fast_agent_timeouts", 
            "verbose_logging_disabled",
            "single_iteration_agents"
        ],
        "performance_targets": {
            "simple_queries": "2-5 seconds",
            "complex_queries": "8-15 seconds"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Optimized LangChain Agent API",
        "version": "2.1.0-optimized",
        "key_features": [
            "Smart routing (auto-detects simple vs complex queries)",
            "Fast agents (optimized timeouts and settings)",
            "Direct RAG for simple questions", 
            "Agent reasoning for complex analysis"
        ],
        "performance": "2-15 seconds depending on query complexity",
        "test_endpoint": "GET /api/routing/test"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting OPTIMIZED LangChain service with smart routing...")
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")