from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
import asyncio
import json
import traceback

app = FastAPI()

# === Ollama LLM Setup ===
llm = ChatOllama(
    model="mistral",
    base_url="http://ollama:11434",
    temperature=0.7
)

# === Debug-Wrapped LLM using RunnableLambda ===
def log_and_invoke(prompt, **kwargs):
    print("🟢 [SYNC] Prompt Sent:", prompt)
    return llm.invoke(prompt, **kwargs)

async def log_and_ainvoke(prompt, **kwargs):
    print("🟢 [ASYNC] Prompt Sent:", prompt)
    return await llm.ainvoke(prompt, **kwargs)

debug_llm = RunnableLambda(func=log_and_invoke, afunc=log_and_ainvoke)

# === Tool: Table to CSV (debug-enabled) ===
def export_table_to_csv(text: str) -> str:
    print("🛠 Tool input:\n", text)
    if "|" in text:
        return (
            "🔍 I detected tabular content.\n"
            "Would you like to turn this into a downloadable CSV file?\n"
            "If yes, I’ll forward it to our internal CSV tool. (Currently under construction.)"
        )
    return "No table format (|) found — skipping CSV generation."

csv_tool = Tool(
    name="export_table_to_csv",
    func=export_table_to_csv,
    description="Use when input contains tables (pipe-formatted like | col1 | col2 |)."
)

# === LangChain AgentExecutor ===
agent_executor = initialize_agent(
    tools=[csv_tool],
    llm=debug_llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# === OpenAI-compatible Output Format ===
def format_response(content: str):
    return {
        "id": "chatcmpl-custom",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content}
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

# === FastAPI Chat Endpoint for LibreChat ===
@app.post("/v1/chat/completions")
async def chat_handler(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        print(f"❌ JSON parse error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON format")

    print("\n=== Incoming Request ===")
    print(json.dumps(body, indent=2))
    print("========================")

    messages = body.get("messages", [])
    stream = body.get("stream", False)

    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="Missing or invalid 'messages'")

    user_prompt = messages[-1].get("content", "")
    print(f"📦 Prompt: {user_prompt} | stream={stream}")

    # === Streaming Response Mode ===
    if stream:
        async def token_stream():
            try:
                async for step in agent_executor.astream({"input": user_prompt}):
                    content = step.get("output", "")
                    if content:
                        payload = {
                            "id": "chatcmpl-stream",
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "delta": {"content": content},
                                "index": 0,
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        await asyncio.sleep(0.01)
                yield "data: [DONE]\n\n"
            except Exception as e:
                print("🔥 Streaming exception:", e)
                traceback.print_exc()
                yield f"data: [ERROR] {str(e)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    # === Non-streaming Response Mode ===
    try:
        result = await agent_executor.ainvoke({"input": user_prompt})
        content = result.get("output", result)
    except Exception as e:
        print("🔥 Non-stream error:", e)
        traceback.print_exc()
        content = f"⚠️ Agent error: {str(e)}"

    return JSONResponse(content=format_response(content))
