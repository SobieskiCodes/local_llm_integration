# main.py  ────────────────────────────────────────────────────────────────
import os, asyncio, json, uuid, logging
from typing import List, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from langchain_ollama import ChatOllama
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain_core.tools import tool
from langchain import hub

# ───────────────────── 0.  GLOBAL ENV / LOGGING ─────────────────────────
# Kill the noisy LangSmith banner
os.environ["LANGCHAIN_TRACING_V2"] = "false"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gateway")

# ───────────────────── 1.  TOOL DEFINITION ──────────────────────────────
@tool
def get_weather(location: str) -> str:
    """Return a (fake) weather report for the given location."""
    return f"It's always sunny in {location}."

# ───────────────────── 2.  MODEL + AGENT SETUP ───────────────────────────
OLLAMA_BASE  = "http://ollama:11434"   # change if host/port differ
OLLAMA_MODEL = "openchat"              # e.g.  'openchat:7b-v3.5'

llm    = ChatOllama(base_url=OLLAMA_BASE, model=OLLAMA_MODEL)
prompt = hub.pull("hwchase17/react")   # canonical ReAct prompt

agent_runnable = create_react_agent(llm, [get_weather], prompt)

agent = AgentExecutor(
    agent=agent_runnable,
    tools=[get_weather],
    verbose=True,                 # prints internal chain steps to console
    handle_parsing_errors=True,   # retry on bad-format output
    max_iterations=3
)

# ───────────────────── 3.  FASTAPI SERVICE ───────────────────────────────
app = FastAPI(title="Ollama-ReAct gateway")

def _format_openai_response(answer: str) -> Dict[str, Any]:
    """Wrap text in OpenAI-style JSON structure expected by LibreChat."""
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "model": OLLAMA_MODEL,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": answer },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
    }

def run_agent(user_input: str) -> str:
    """
    Invoke the agent with retries; fall back to raw LLM reply
    if parsing still fails after AgentExecutor’s automatic retries.
    """
    try:
        out = agent.invoke({"input": user_input})
        return out["output"] if isinstance(out, dict) else str(out)
    except Exception as e:
        logger.exception("❌ Agent failed – falling back to raw LLM. Reason: %s", e)
        raw = llm.invoke(user_input)
        return (
            "⚠️ _Agent formatting failed – showing raw LLM reply_\n\n"
            + (getattr(raw, "content", str(raw)))
        )

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    msgs: List[Dict[str, str]] = body.get("messages", [])
    if not msgs:
        raise HTTPException(400, "Missing 'messages' array")

    # LibreChat sends the whole convo; grab the newest user turn
    user_input = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
    if not user_input:
        raise HTTPException(400, "No user message found")

    stream = bool(body.get("stream", False))

    if not stream:
        answer_text = run_agent(user_input)
        return JSONResponse(_format_openai_response(answer_text))

    # ───── streaming branch (Server-Sent Events) ──────────────────────
    async def event_stream():
        txt = run_agent(user_input)
        for token in txt.split():
            chunk = {
                "id": None,
                "object": "chat.completion.chunk",
                "model": OLLAMA_MODEL,
                "choices": [{
                    "index": 0,
                    "delta": { "content": token + " " },
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0)

        # final stop chunk
        yield (
            "data: "
            + json.dumps({
                "id": None,
                "object": "chat.completion.chunk",
                "model": OLLAMA_MODEL,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            })
            + "\n\n"
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/health")
def health():
    return {"status": "ok"}
