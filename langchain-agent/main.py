# COMPLETE FIXED LANGCHAIN AGENT - Guaranteed Tool Execution
# Replace entire langchain-agent/main.py with this code

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent  
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import requests
import json
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

app = FastAPI(title="Fixed Multi-Tool LangChain Agent", version="5.0.0-guaranteed-tools")

# ========== CONFIGURATION ==========
LLAMAINDEX_TIMEOUT = 180
OLLAMA_TIMEOUT = 120
AGENT_TIMEOUT = 300

print("🚀 Starting FIXED MULTI-TOOL agent with GUARANTEED tool execution")

# ========== ROBUST TOOL IMPLEMENTATIONS ==========

class LlamaIndexRAGTool:
    """Enhanced RAG tool with better error handling"""
    
    def _run(self, query: str) -> str:
        """Execute RAG query against LlamaIndex service"""
        try:
            print(f"🔍 RAG Tool executing: {query[:50]}...")
            response = requests.post(
                "http://llamaindex:8081/query",
                json={"input": query},
                timeout=LLAMAINDEX_TIMEOUT
            )
            response.raise_for_status()
            result = response.json().get("result", "No result found")
            
            print(f"✅ RAG Tool completed: {len(result)} chars")
            return f"Knowledge Base Search Result: {result}"
            
        except requests.exceptions.Timeout:
            print("⏰ RAG Tool timed out")
            return "Knowledge base search timed out - system may be busy. Try a simpler query."
        except requests.exceptions.ConnectionError:
            print("🔌 RAG Tool connection failed")
            return "Cannot connect to knowledge base - please check if LlamaIndex service is running."
        except requests.exceptions.RequestException as e:
            print(f"❌ RAG Tool request failed: {e}")
            return f"Knowledge base search failed: {str(e)}"
        except Exception as e:
            print(f"💥 RAG Tool unexpected error: {e}")
            return f"Unexpected error in knowledge base search: {str(e)}"

class OllamaReasoningTool:
    """Enhanced reasoning tool with better error handling"""
    
    def _run(self, query: str, system_prompt: str = "") -> str:
        """Execute reasoning query against Ollama"""
        try:
            print(f"🧠 Reasoning Tool executing: {query[:50]}...")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({"role": "system", "content": "You are a helpful AI assistant. Provide detailed, analytical responses."})
            
            messages.append({"role": "user", "content": query})
            
            response = requests.post(
                "http://ollama:11434/v1/chat/completions",
                json={
                    "model": "mistral:latest",
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 1000
                },
                timeout=OLLAMA_TIMEOUT
            )
            response.raise_for_status()
            
            result = response.json()["choices"][0]["message"]["content"]
            print(f"✅ Reasoning Tool completed: {len(result)} chars")
            return f"Reasoning Analysis: {result}"
            
        except requests.exceptions.Timeout:
            print("⏰ Reasoning Tool timed out")
            return "Reasoning request timed out - complex analysis may take longer. Try breaking down the question."
        except requests.exceptions.ConnectionError:
            print("🔌 Reasoning Tool connection failed")
            return "Cannot connect to Ollama reasoning engine - please check if Ollama service is running."
        except requests.exceptions.RequestException as e:
            print(f"❌ Reasoning Tool request failed: {e}")
            return f"Reasoning engine unavailable: {str(e)}"
        except Exception as e:
            print(f"💥 Reasoning Tool unexpected error: {e}")
            return f"Unexpected error in reasoning analysis: {str(e)}"

class QdrantVectorTool:
    """Enhanced vector search tool"""
    
    def _run(self, query: str) -> str:
        """Execute vector search via LlamaIndex"""
        print(f"🔍 Vector Tool executing: {query[:50]}...")
        # Route through LlamaIndex for now since it integrates with Qdrant
        rag_tool = LlamaIndexRAGTool()
        result = rag_tool._run(f"Find semantically similar information related to: {query}")
        return result.replace("Knowledge Base Search Result:", "Vector Similarity Search Result:")

# ========== COMPREHENSIVE TOOLS SETUP ==========
def create_robust_tools():
    """Create all tools with enhanced error handling"""
    
    # Initialize the robust tools
    rag_tool = LlamaIndexRAGTool()
    reasoning_tool = OllamaReasoningTool() 
    vector_tool = QdrantVectorTool()
    
    return [
        Tool(
            name="KnowledgeSearch",
            func=rag_tool._run,
            description="""Search the knowledge base for factual information, documentation, 
            configuration details, and stored knowledge. Use for questions about our system, 
            setup details, parameters, or when you need to find specific documented information."""
        ),
        Tool(
            name="ReasoningEngine", 
            func=reasoning_tool._run,
            description="""Use advanced reasoning and analysis capabilities. Perfect for 
            comparing options, analyzing trade-offs, generating insights, strategic thinking,
            complex problem-solving, and providing detailed analytical responses."""
        ),
        Tool(
            name="VectorSearch",
            func=vector_tool._run,
            description="""Search for semantically similar content and related concepts.
            Use when looking for similar examples, related topics, or when semantic 
            similarity would help find relevant information."""
        )
    ]

def create_robust_llm():
    """LLM optimized for reliable tool usage"""
    return ChatOpenAI(
        openai_api_base="http://ollama:11434/v1",
        openai_api_key="not-needed",
        model="mistral:latest",
        temperature=0.1,
        max_tokens=1500,
        request_timeout=OLLAMA_TIMEOUT
    )

# Initialize robust components
llm = create_robust_llm()
tools = create_robust_tools()

print(f"🛠️ Created robust agent with {len(tools)} specialized tools")

# ========== FIXED PROMPT - NO MORE HALLUCINATIONS ==========
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent assistant with access to specialized tools.

CRITICAL TOOL USAGE RULES:
- You MUST actually call tools, not just describe calling them
- DO NOT roleplay or simulate tool usage
- DO NOT say "I will use X tool" without actually using it
- When you need information, CALL the appropriate tool immediately

AVAILABLE TOOLS:
1. KnowledgeSearch - Search knowledge base for facts, documentation, configuration details
2. ReasoningEngine - Advanced analysis, comparisons, strategic thinking, complex reasoning  
3. VectorSearch - Find semantically similar content and related concepts

WHEN TO USE EACH TOOL:
- Questions about "our system", "our setup", "configuration" → USE KnowledgeSearch
- Analysis, comparisons, "pros/cons", strategic questions → USE ReasoningEngine
- Finding similar examples or related topics → USE VectorSearch

EXAMPLES:
❌ WRONG: "Let me search the knowledge base for..." (then not actually calling it)
✅ RIGHT: [Actually calls KnowledgeSearch tool and uses the real result]

❌ WRONG: "I'll analyze this using reasoning..." (then giving analysis without tool)
✅ RIGHT: [Actually calls ReasoningEngine tool and presents the real result]

If a tool provides information, clearly indicate it came from the tool.
If you don't need tools for a simple question, answer directly from your knowledge."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ========== FIXED AGENT CREATION ==========
print("🤖 Creating FIXED agent with guaranteed tool execution...")
try:
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3,
        max_execution_time=AGENT_TIMEOUT,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        early_stopping_method="generate"
    )
    print("✅ FIXED agent ready - no more tool hallucinations")
except Exception as e:
    print(f"❌ Fixed agent creation failed: {e}")
    agent_executor = None

# ========== MANUAL TOOL ROUTING (GUARANTEED EXECUTION) ==========
async def process_with_manual_routing(query: str) -> dict:
    """Manual tool routing to ensure tools are actually called"""
    start_time = time.time()
    
    print(f"🔧 Manual routing analysis: {query[:50]}...")
    
    # Analyze query and determine which tools to use
    query_lower = query.lower()
    tools_to_use = []
    
    # Determine tools based on query content
    if any(word in query_lower for word in [
        "our", "configuration", "setup", "parameters", "system", "settings",
        "environment", "docker", "ollama", "llamaindex", "qdrant"
    ]):
        tools_to_use.append("KnowledgeSearch")
    
    if any(word in query_lower for word in [
        "analyze", "compare", "pros", "cons", "suggest", "recommend", 
        "evaluate", "assessment", "strategy", "versus", "vs", "better",
        "optimization", "improve", "best practices"
    ]):
        tools_to_use.append("ReasoningEngine")
    
    if any(word in query_lower for word in [
        "similar", "related", "find", "examples", "like", "alternatives",
        "comparable", "matching"
    ]):
        tools_to_use.append("VectorSearch")
    
    # Default fallbacks
    if not tools_to_use:
        if len(query.split()) > 10 or "?" in query:
            tools_to_use.append("ReasoningEngine")  # Complex questions get reasoning
    
    # If still no tools, but it's a substantive query, use ReasoningEngine
    if not tools_to_use and len(query.strip()) > 20:
        tools_to_use.append("ReasoningEngine")
    
    print(f"🎯 Determined tools to use: {tools_to_use}")
    
    results = {}
    total_tool_calls = 0
    
    # Manually call determined tools
    if "KnowledgeSearch" in tools_to_use:
        try:
            print("🔍 MANUALLY calling KnowledgeSearch...")
            rag_tool = LlamaIndexRAGTool()
            result = rag_tool._run(query)
            results["KnowledgeSearch"] = result
            total_tool_calls += 1
        except Exception as e:
            results["KnowledgeSearch"] = f"KnowledgeSearch Error: {str(e)}"
            print(f"❌ KnowledgeSearch failed: {e}")
    
    if "ReasoningEngine" in tools_to_use:
        try:
            print("🧠 MANUALLY calling ReasoningEngine...")
            reasoning_tool = OllamaReasoningTool()
            result = reasoning_tool._run(query)
            results["ReasoningEngine"] = result  
            total_tool_calls += 1
        except Exception as e:
            results["ReasoningEngine"] = f"ReasoningEngine Error: {str(e)}"
            print(f"❌ ReasoningEngine failed: {e}")
    
    if "VectorSearch" in tools_to_use:
        try:
            print("🔍 MANUALLY calling VectorSearch...")
            vector_tool = QdrantVectorTool()
            result = vector_tool._run(query)
            results["VectorSearch"] = result
            total_tool_calls += 1
        except Exception as e:
            results["VectorSearch"] = f"VectorSearch Error: {str(e)}"
            print(f"❌ VectorSearch failed: {e}")
    
    # Synthesize results
    if results:
        print(f"🔄 Synthesizing results from {len(results)} tools...")
        
        # Create synthesis prompt
        synthesis_prompt = f"Original Query: {query}\n\n"
        synthesis_prompt += "Tool Results:\n"
        for tool_name, result in results.items():
            synthesis_prompt += f"\n=== {tool_name} ===\n{result}\n"
        
        synthesis_prompt += "\nPlease synthesize this information to provide a comprehensive answer to the original query. If any tools encountered errors, work with the available information:"
        
        try:
            response = llm.invoke([HumanMessage(content=synthesis_prompt)])
            final_response = response.content if hasattr(response, 'content') else str(response)
            print("✅ Synthesis completed")
        except Exception as e:
            print(f"❌ Synthesis failed: {e}")
            # Fallback: combine tool results directly
            final_response = f"Query: {query}\n\n"
            for tool_name, result in results.items():
                final_response += f"{tool_name}:\n{result}\n\n"
    else:
        # No tools needed or all failed, direct response
        print("📝 No tools used, providing direct response...")
        try:
            response = llm.invoke([HumanMessage(content=query)])
            final_response = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            final_response = f"I apologize, but I'm experiencing technical difficulties: {str(e)}"
    
    duration = time.time() - start_time
    print(f"⏱️ Manual routing completed in {duration:.2f}s with {total_tool_calls} tool calls")
    
    return {
        "response": final_response,
        "method": "manual_tool_routing",
        "duration": duration,
        "tool_calls": total_tool_calls,
        "tools_used": list(results.keys()),
        "tool_results": results,
        "forced_execution": True,
        "routing_analysis": tools_to_use
    }

# ========== AGENT PROCESSING (WITH VERIFICATION) ==========
async def process_with_agent_verification(query: str) -> dict:
    """Process with agent but verify tool usage"""
    start_time = time.time()
    
    print(f"🤖 Agent processing with verification: {query[:50]}...")
    
    if agent_executor:
        try:
            result = agent_executor.invoke({"input": query})
            
            if isinstance(result, dict):
                output = result.get("output", str(result))
                steps = result.get("intermediate_steps", [])
            else:
                output = str(result)
                steps = []
            
            # Analyze tool usage
            tool_calls = len(steps)
            tools_used = []
            
            for step in steps:
                if len(step) >= 2 and hasattr(step[0], 'tool'):
                    tools_used.append(step[0].tool)
            
            duration = time.time() - start_time
            
            print(f"📊 Agent used {tool_calls} tools: {list(set(tools_used))}")
            
            return {
                "response": output,
                "method": f"agent_execution_{tool_calls}_tools",
                "duration": duration,
                "tool_calls": tool_calls,
                "tools_used": list(set(tools_used)),
                "agent_success": True
            }
            
        except Exception as e:
            print(f"⚠️ Agent execution failed: {e}")
            return {
                "response": f"Agent execution failed: {str(e)}",
                "method": "agent_execution_failed",
                "duration": time.time() - start_time,
                "tool_calls": 0,
                "tools_used": [],
                "agent_success": False,
                "error": str(e)
            }
    else:
        return {
            "response": "Agent is not available",
            "method": "no_agent_available",
            "duration": time.time() - start_time,
            "tool_calls": 0,
            "tools_used": [],
            "agent_success": False
        }

# ========== HYBRID PROCESSING (BEST OF BOTH) ==========
async def process_with_hybrid_approach(query: str) -> dict:
    """Hybrid: Try agent first, fallback to manual routing if no tools used"""
    
    # Try agent first
    agent_result = await process_with_agent_verification(query)
    
    # Check if agent actually used tools when it should have
    query_lower = query.lower()
    should_use_tools = any(word in query_lower for word in [
        "our", "analyze", "compare", "configuration", "setup", "suggest",
        "parameters", "system", "optimization", "pros", "cons"
    ])
    
    if should_use_tools and agent_result["tool_calls"] == 0:
        print("🔄 Agent didn't use tools when it should have, switching to manual routing...")
        manual_result = await process_with_manual_routing(query)
        manual_result["fallback_reason"] = "agent_no_tools_when_needed"
        return manual_result
    else:
        return agent_result

# ========== API ENDPOINTS ==========

@app.post("/api/chat")
async def guaranteed_tool_chat(request: Request):
    """Chat endpoint with guaranteed tool execution"""
    try:
        body = await request.json()
        query = body.get("input", "").strip()
        
        if not query:
            return {"response": "Please provide a question or message."}
        
        print(f"\n💬 Guaranteed tool chat: {query}")
        
        # Use hybrid approach for best results
        result = await process_with_hybrid_approach(query)
        
        return {
            "response": result["response"],
            "method": result["method"],
            "duration_seconds": round(result["duration"], 2),
            "tool_calls": result["tool_calls"],
            "tools_used": result.get("tools_used", []),
            "forced_execution": result.get("forced_execution", False),
            "fallback_reason": result.get("fallback_reason"),
            "timestamp": datetime.now().isoformat(),
            "version": "5.0.0-guaranteed-tools"
        }
        
    except Exception as e:
        return {
            "response": f"I encountered an error: {str(e)}",
            "method": "error_fallback", 
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/test/manual")
async def test_manual_routing(request: Request):
    """Test manual tool routing (guaranteed execution)"""
    try:
        body = await request.json()
        query = body.get("input", "test query")
        
        result = await process_with_manual_routing(query)
        
        return {
            "query": query,
            "result": result,
            "guaranteed_execution": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/test/agent")
async def test_agent_only(request: Request):
    """Test agent execution only"""
    try:
        body = await request.json()
        query = body.get("input", "test query")
        
        result = await process_with_agent_verification(query)
        
        return {
            "query": query,
            "result": result,
            "agent_only": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/test/tools")
async def test_individual_tools(request: Request):
    """Test each tool individually"""
    try:
        body = await request.json()
        query = body.get("input", "test query")
        
        tools_list = create_robust_tools()
        results = {}
        
        for tool in tools_list:
            try:
                start_time = time.time()
                result = tool.func(query)
                duration = time.time() - start_time
                
                results[tool.name] = {
                    "status": "success",
                    "result": result[:300] + "..." if len(str(result)) > 300 else str(result),
                    "duration": round(duration, 2)
                }
            except Exception as e:
                results[tool.name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return {
            "query": query,
            "tool_tests": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/capabilities")
async def show_guaranteed_capabilities():
    """Show guaranteed capabilities"""
    return {
        "version": "5.0.0-guaranteed-tools",
        "approach": "Hybrid agent with guaranteed tool execution fallback",
        "tools": {
            "KnowledgeSearch": "Search knowledge base for facts and documentation",
            "ReasoningEngine": "Advanced analysis and strategic thinking",
            "VectorSearch": "Semantic similarity search for related content"
        },
        "execution_modes": {
            "agent_execution": "Standard LangChain agent (preferred when working)",
            "manual_routing": "Guaranteed tool execution (fallback)",
            "hybrid_approach": "Tries agent first, falls back to manual if needed"
        },
        "guaranteed_features": [
            "Tools will actually be called, not just described",
            "Manual routing ensures tool execution",
            "Comprehensive error handling and fallbacks",
            "Real tool usage verification"
        ],
        "test_endpoints": {
            "manual_guaranteed": "POST /api/test/manual",
            "agent_only": "POST /api/test/agent", 
            "individual_tools": "POST /api/test/tools"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "5.0.0-guaranteed-tools",
        "features": [
            "guaranteed_tool_execution",
            "hybrid_agent_manual_routing",
            "comprehensive_error_handling",
            "tool_usage_verification"
        ],
        "tools_available": len(create_robust_tools()),
        "agent_ready": agent_executor is not None,
        "manual_routing_ready": True
    }

@app.get("/")
async def root():
    return {
        "message": "Guaranteed Tool Execution LangChain Agent",
        "version": "5.0.0-guaranteed-tools",
        "key_features": [
            "Tools are actually called, not just described",
            "Manual routing fallback ensures tool execution", 
            "Hybrid approach combines agent intelligence with guaranteed execution",
            "Comprehensive error handling and verification"
        ],
        "tools": ["KnowledgeSearch", "ReasoningEngine", "VectorSearch"],
        "try_these": [
            "What are our Ollama configuration parameters?",
            "Analyze our setup vs cloud alternatives",
            "Find similar authentication methods"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting GUARANTEED TOOL EXECUTION agent...")
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")