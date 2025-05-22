from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_ollama import ChatOllama
from langchain.agents import Tool, initialize_agent, AgentType
from langgraph.graph import StateGraph, END
from typing import TypedDict
import json
import asyncio
import traceback
import uuid
import time
import logging

# === Logging setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === FastAPI app ===
app = FastAPI()

# === LLM setup ===
llm = ChatOllama(
    model="mistral",
    base_url="http://ollama:11434",
    temperature=0.7
)

# === Minimal dummy tool to satisfy LangChain

def noop_tool(_):
    return "This tool is not in use."

tools = [
    Tool(
        name="NoopTool",
        func=noop_tool,
        description="This tool does nothing and exists only to initialize the agent."
    )
]

# === State typing ===
class AgentState(TypedDict):
    input: str
    output: str

# === Initialize ReAct agent with dummy tool
react_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True  # ✅ Enable verbose logging for full agent steps
)

# === LangGraph node
async def agent_node(state: AgentState) -> AgentState:
    input_text = state["input"]
    print(f"[agent_node] Running agent on: {input_text}")
    try:
        result = await react_agent.ainvoke(input_text)
        print(f"[agent_node] Raw agent result: {result}")
        return {"input": input_text, "output": result}
    except Exception as e:
        traceback.print_exc()
        return {"input": input_text, "output": f"Agent error: {str(e)}"}

# === LangGraph setup
workflow = StateGraph(AgentState)
workflow.add_node("agent_node", agent_node)
workflow.set_entry_point("agent_node")
workflow.add_edge("agent_node", END)
graph = workflow.compile()

# === FastAPI endpoint for LibreChat
@app.post("/v1/chat/completions")
async def chat_handler(request: Request):
    try:
        body = await request.json()
        print("[chat_handler] Received body:", json.dumps(body))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON format")

    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="Missing or invalid 'messages'")

    user_prompt = messages[-1].get("content", "")
    input_data = {"input": user_prompt}

    try:
        result = await graph.ainvoke(input_data)
        print("[chat_handler] Raw graph result:", json.dumps(result, indent=2))

        raw_output = result.get("output", "")
        if isinstance(raw_output, dict):
            content = raw_output.get("output", json.dumps(raw_output))
        else:
            content = str(raw_output) or "Sorry, I couldn't generate a response."

        print("[chat_handler] Final content:", content)
    except Exception as e:
        traceback.print_exc()
        content = f"Agent error: {str(e)}"

    # === Simulate OpenAI-style streaming
    async def token_stream():
        resp_id = f"chatcmpl-{uuid.uuid4().hex}"
        model_name = "mistral"
        created_time = int(time.time())

        first_chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [{
                "delta": {"role": "assistant"},
                "index": 0,
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"

        for i, word in enumerate(content.split()):
            chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [{
                    "delta": {"content": " " + word if i else word},
                    "index": 0,
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)

        final_chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [{
                "delta": {},
                "index": 0,
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")