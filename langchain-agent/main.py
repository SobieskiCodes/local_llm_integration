"""
FastAPI → LangGraph ReAct Agent → Ollama (openchat)
Console-first edition: full live chain output
"""

import os, asyncio, json, uuid, logging
from typing import List, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from langchain_ollama import ChatOllama
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain_core.tools import tool
from langchain.callbacks.stdout import StreamingStdOutCallbackHandler
from langchain import hub

# ───────────── 0.  ENV & LOGGING ────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "false"   # silence LangSmith banner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
root_log = logging.getLogger()
root_log.setLevel(logging.INFO)

# ───────────── 1.  TOOL EXAMPLE ─────────────────────────────────────────
@tool
def get_weather(location: str) -> str:
    """Return a (fake) weather report for the given location."""
    return f"It's always sunny in {location}."

# ───────────── 2.  MODEL & AGENT SETUP ──────────────────────────────────
OLLAMA_BASE  = "http://ollama:11434"
OLLAMA_MODEL = "openchat"          # change to openchat:7b-v3.5 etc.

llm    = ChatOllama(base_url=OLLAMA_BASE, model=OLLAMA_MODEL, verbose=True)
prompt = hub.pull("hwchase17/react")

agent_runnable = create_react_agent(llm, [get_weather], prompt)

agent = AgentExecutor(
    agent=agent_runnable,
    tools=[get_weather],
    verbose=True,  # <- prints every step automatically
    handle_parsing_errors=True,
    max_iterations=3,
)

# ───────────── 3.  FASTAPI SERVICE ──────────────────────────────────────
app = FastAPI(title="Ollama-ReAct gateway (console-first)")

def wrap_openai(answer: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "model": OLLAMA_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

def run_agent(user_prompt: str) -> str:
    """Run agent; fallback to raw LLM if it refuses or crashes."""
    try:
        result = agent.invoke(
            {"input": user_prompt},
            callbacks=[StreamingStdOutCallbackHandler()],   # <- live prints
        )
        text = result["output"] if isinstance(result, dict) else str(result)

        if any(k in text.lower() for k in ("unable to answer", "cannot")):
            root_log.info("Agent declined – retrying with direct LLM")
            text = llm.invoke(user_prompt).content
        return text

    except Exception as e:
        root_log.exception("Agent error, using raw LLM instead: %s", e)
        return (
            "⚠️ _Agent failed – showing raw LLM reply_\n\n"
            + llm.invoke(user_prompt).content
        )

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    root_log.info("📥 Incoming body: %s", json.dumps(body, indent=2)[:1000])

    messages: List[Dict[str, str]] = body.get("messages", [])
    if not messages:
        raise HTTPException(400, "Missing 'messages' array")

    user_prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if not user_prompt:
        raise HTTPException(400, "No user message")

    stream = bool(body.get("stream", False))
    answer_text = await asyncio.get_event_loop().run_in_executor(None, lambda: run_agent(user_prompt))

    if not stream:
        return JSONResponse(wrap_openai(answer_text))

    # simple SSE stream (final answer only)
    async def event_stream():
        for token in answer_text.split():
            chunk = {
                "id": None,
                "object": "chat.completion.chunk",
                "model": OLLAMA_MODEL,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token + " "},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0)

        yield (
            "data: "
            + json.dumps({
                "id": None,
                "object": "chat.completion.chunk",
                "model": OLLAMA_MODEL,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            })
            + "\n\n"
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/health")
def health():
    return {"status": "ok"}

# === Model-list endpoint for LibreChat ===
@app.get("/v1/models")
def list_models():
    """
    Minimal OpenAI-compatible model list.
    LibreChat calls this when `fetch: true`.
    """
    models = [
        {"id": "openchat", "object": "model", "owned_by": "local"},
        {"id": "mistral",  "object": "model", "owned_by": "local"},
    ]
    return {"object": "list", "data": models}
