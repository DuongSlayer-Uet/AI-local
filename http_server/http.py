from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3:8b"
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Do not use thinking mode. "
    "Answer directly and concisely in 2-3 sentences. Avoid unnecessary explanations."
)

app = FastAPI(
    title="AI Local Server",
    description="REST API for Raspberry Pi AI Client",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/api/v1/health")
async def health_check():
    return {"status": "online", "service": "AI Local Server", "version": "1.0.0", "timestamp": now()}


@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.message},
        ],
        "think": False,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, "Cannot connect to Ollama")
    except httpx.TimeoutException:
        raise HTTPException(504, "LLM inference timeout")
    except httpx.HTTPStatusError as error:
        raise HTTPException(502, f"Ollama error: {error.response.text}")

    result = response.json()
    return {
        "status": "success",
        "model": MODEL_NAME,
        "response": result["message"]["content"],
        "timestamp": now(),
    }