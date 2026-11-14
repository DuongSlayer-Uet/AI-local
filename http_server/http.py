import tempfile
from datetime import datetime, timezone

import httpx
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3:8b"
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Do not use thinking mode. "
    "Answer directly and concisely in 2-3 sentences. Avoid unnecessary explanations."
)

WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
WHISPER_LANGUAGE = "vietnamese"  # đổi hoặc để None nếu muốn Whisper tự detect ngôn ngữ
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Whisper một lần duy nhất khi server khởi động, không load lại mỗi request
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_MODEL_ID, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
)
whisper_model.to(device)
whisper_model.generation_config.forced_decoder_ids = None  # dùng language/task thay vì config cũ

whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID, clean_up_tokenization_spaces=False)

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    dtype=dtype,
    device=device,
    generate_kwargs={"language": WHISPER_LANGUAGE, "task": "transcribe"},
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


async def ask_ollama(message: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
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

    return response.json()["message"]["content"]


@app.get("/api/v1/health")
async def health_check():
    return {"status": "online", "service": "AI Local Server", "version": "1.0.0", "timestamp": now()}


@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    reply = await ask_ollama(request.message)
    return {"status": "success", "model": MODEL_NAME, "response": reply, "timestamp": now()}


@app.post("/api/v1/audio")
async def audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Only .wav files are supported")

    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        transcript = asr_pipe(tmp.name)["text"]

    reply = await ask_ollama(transcript)
    return {
        "status": "success",
        "model": MODEL_NAME,
        "transcript": transcript,
        "response": reply,
        "timestamp": now(),
    }