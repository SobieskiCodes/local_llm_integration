from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import json
import datetime
import requests

app = FastAPI()

# === Utility: Format chat history ===
def format_chat_history(messages):
    history = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            history.append(f"USER: {content}")
        elif role == "assistant":
            history.append(f"ASSISTANT: {content}")
    return "\n".join(history)

# === POST /api/chat ===
@app.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        print("\n/api/chat received raw body:", json.dumps(body))

        messages = body.get("messages", [])
        stream = body.get("stream", False)
        model = body.get("model", "mistral")
        print("stream flag:", stream)

        if not messages or not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="Missing or invalid 'messages'")

        raw_chat = format_chat_history(messages[:-1])
        if "<chat_history>" in messages[-1]["content"]:
            messages[-1]["content"] = messages[-1]["content"].replace("<chat_history>", raw_chat)

        if stream:
            async def proxy_stream():
                first_chunk = True
                try:
                    async with httpx.AsyncClient(timeout=None) as client:
                        async with client.stream(
                            "POST", "http://ollama:11434/api/chat",
                            json={"model": model, "messages": messages, "stream": True}
                        ) as resp:
                            async for line in resp.aiter_lines():
                                if not line.strip():
                                    continue
                                if line.startswith("data: "):
                                    raw = line.removeprefix("data: ").strip()
                                else:
                                    raw = line.strip()

                                if raw == "[DONE]":
                                    print("✅ Reached [DONE] marker")
                                    yield "data: [DONE]\n\n"
                                    break

                                try:
                                    parsed = json.loads(raw)
                                    token = parsed.get("message", {}).get("content", "")
                                    if token:
                                        chunk = {
                                            "id": "stream",
                                            "object": "chat.completion.chunk",
                                            "choices": [{
                                                "delta": {"content": token},
                                                "index": 0,
                                                "finish_reason": None
                                            }]
                                        }
                                        print("⬅️  sending chunk:", token)
                                        yield f"data: {json.dumps(chunk)}\n\n"
                                except Exception as e:
                                    print("Stream parse error:", e)
                except Exception as e:
                    print("🔥 Stream failed:", str(e))

            return StreamingResponse(proxy_stream(), media_type="text/event-stream")

        # === Non-streaming fallback ===
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://ollama:11434/api/chat",
                json={"model": model, "messages": messages, "stream": False}
            )

            content = await response.aread()
            print("Raw Ollama response (bytes):", content)

            try:
                data = json.loads(content)
            except Exception as e:
                print("JSON parse failed:", e)
                print("Raw Ollama content decoded:\n", content.decode("utf-8", errors="ignore"))
                raise HTTPException(status_code=500, detail="Failed to parse Ollama response as JSON")

            if "message" not in data:
                print("Unexpected response format from Ollama:", data)
                raise HTTPException(status_code=502, detail="Invalid format from LLM")

            output_text = data["message"].get("content", "")

            return {
                "id": "chatcmpl-001",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text
                    },
                    "finish_reason": "stop"
                }],
                "created": int(datetime.datetime.now().timestamp()),
                "model": model,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

    except Exception as e:
        print("Error in /api/chat:", str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    try:
        body = await request.json()
        print("[v1/chat/completions] Received:", json.dumps(body))

        messages = body.get("messages", [])
        model = body.get("model", "mistral")
        stream = body.get("stream", False)

        if not messages:
            raise HTTPException(status_code=400, detail="Missing messages")

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        if stream:
            async def stream_response():
                try:
                    async with httpx.AsyncClient(timeout=None) as client:
                        async with client.stream(
                            "POST", "http://localhost:8082/api/chat",
                            json=payload
                        ) as resp:
                            async for line in resp.aiter_lines():
                                if not line.strip():
                                    continue
                                if line.startswith("data: "):
                                    yield line + "\n\n"
                                else:
                                    yield f"data: {line}\n\n"
                except Exception as e:
                    print("Streaming error:", str(e))
                    yield f"data: [ERROR] {str(e)}\n\n"

            return StreamingResponse(stream_response(), media_type="text/event-stream")

        else:
            async with httpx.AsyncClient() as client:
                resp = await client.post("http://localhost:8082/api/chat", json=payload)
                return JSONResponse(content=resp.json())

    except Exception as e:
        print("Parse error:", str(e))
        raise HTTPException(status_code=400, detail="Invalid JSON")


# === GET /api/tags ===
@app.get("/api/tags")
async def proxy_tags():
    try:
        res = requests.get("http://ollama:11434/api/tags")
        res.raise_for_status()
        return JSONResponse(content=res.json())
    except Exception as e:
        print("/api/tags error:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# === GET /api/version ===
@app.get("/api/version")
def api_version():
    return {
        "version": "0.9.0-debug",
        "component": "Streaming Ollama Proxy with Enhanced Debug Logging"
    }