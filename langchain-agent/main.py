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

# 🛠️ Tool declaration with strong description
tool = Tool(
    name="LlamaIndexRAG",
    func=query_llamaindex,
    description=(
        "Use this tool when the user asks a factual, knowledge-based, or research-type question. "
        "Input should be the exact user question as a string."
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
    ("system", "You are a smart assistant. If a tool is available that can answer the user's question, use it."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 🧠 Structured function-calling agent setup
agent = create_openai_functions_agent(llm=llm, tools=[tool], prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True)

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
    return {"version": "0.3.1", "component": "LangChain agent with structured function calling"}
