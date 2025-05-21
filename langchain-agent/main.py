from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests

app = FastAPI()

# 🔧 Tool function to query LlamaIndex API
def query_llamaindex(input_text: str) -> str:
    try:
        res = requests.post("http://llamaindex:8081/query", json={"input": input_text})
        res.raise_for_status()
        return str(res.json().get("result", "No result key in response"))
    except Exception as e:
        return f"Tool call failed: {str(e)}"

# 🔧 Tool function to query CrewAI DSCSA Agent
def query_dscsa_agent(input_text: str) -> str:
    try:
        res = requests.post("http://crewai-agents:8083/query-dscsa", json={"query": input_text})
        res.raise_for_status()
        return str(res.json().get("result", "No result from DSCSA agent"))
    except Exception as e:
        return f"DSCSA agent call failed: {str(e)}"

# 🔧 Tool function to query CrewAI Database Analytics Agent
def query_db_analytics_agent(input_text: str) -> str:
    try:
        res = requests.post("http://crewai-agents:8083/analyze-database", json={"query": input_text})
        res.raise_for_status()
        return str(res.json().get("result", "No result from DB analytics agent"))
    except Exception as e:
        return f"DB analytics agent call failed: {str(e)}"

# 🔧 Tool function to use multi-agent capabilities
def query_multi_agent(input_text: str) -> str:
    try:
        res = requests.post("http://crewai-agents:8083/multi-agent-query", json={"query": input_text})
        res.raise_for_status()
        return str(res.json().get("result", "No result from multi-agent query"))
    except Exception as e:
        return f"Multi-agent query failed: {str(e)}"

# 🛠️ Tool declarations with strong descriptions
llamaindex_tool = Tool(
    name="LlamaIndexRAG",
    func=query_llamaindex,
    description=(
        "Use this tool when the user asks a factual, knowledge-based, or research-type question. "
        "Input should be the exact user question as a string."
    )
)

dscsa_tool = Tool(
    name="DSCSAComplianceAgent",
    func=query_dscsa_agent,
    description=(
        "Use this tool when the user asks about DSCSA (Drug Supply Chain Security Act) compliance, "
        "pharmaceutical regulations, drug tracing requirements, or pharma supply chain security. "
        "Input should be the specific DSCSA-related question as a string."
    )
)

db_analytics_tool = Tool(
    name="DatabaseAnalyticsAgent",
    func=query_db_analytics_agent,
    description=(
        "Use this tool when the user asks for database analysis, SQL queries, or data insights. "
        "Input should be the specific database question or SQL query as a string."
    )
)

multi_agent_tool = Tool(
    name="MultiAgentSystem",
    func=query_multi_agent,
    description=(
        "Use this tool for complex queries that might require coordination between multiple agents "
        "such as combined DSCSA compliance and database analytics tasks. "
        "Input should be the complete question requiring multiple agent skills."
    )
)

# 🤖 LLM connected to local Ollama
llm = ChatOpenAI(
    openai_api_base="http://ollama:11434/v1",
    openai_api_key="not-needed",  # dummy key for compatibility
    model="mistral:latest"
)

# 📋 Structured prompt to nudge proper tool use
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a smart assistant with access to multiple specialized agents.
    
    For general knowledge questions, use the LlamaIndexRAG tool.
    
    For DSCSA (Drug Supply Chain Security Act) compliance questions, pharmaceutical regulations, 
    drug tracing requirements, or pharma supply chain security, use the DSCSAComplianceAgent tool.
    
    For database analysis, SQL queries, or data insights, use the DatabaseAnalyticsAgent tool.
    
    For complex questions that might require multiple skills, use the MultiAgentSystem tool.
    
    Always use the most appropriate tool to give the user the best possible answer."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 🧠 Structured function-calling agent setup
agent = create_openai_functions_agent(
    llm=llm, 
    tools=[llamaindex_tool, dscsa_tool, db_analytics_tool, multi_agent_tool],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=[llamaindex_tool, dscsa_tool, db_analytics_tool, multi_agent_tool], 
    verbose=True
)

# ✅ LangChain-style chat endpoint
@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    query_text = body.get("input")
    result = agent_executor.run(query_text)
    return {"response": result}

# 🧪 OpenAI-compatible chat completion endpoint
@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    input_text = messages[-1]["content"] if messages else ""
    result = agent_executor.run(input_text)
    return {
        "id": "chatcmpl-001",
        "object": "chat.completion",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": result
            },
            "finish_reason": "stop",
            "index": 0
        }],
        "model": "mistral:latest",
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

# 🔧 Debug tag passthrough
@app.get("/api/tags")
async def proxy_tags():
    try:
        res = requests.get("http://ollama:11434/api/tags")
        res.raise_for_status()
        return JSONResponse(content=res.json())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/version")
def api_version():
    return {"version": "0.3.2", "component": "LangChain agent with integrated CrewAI agents"}