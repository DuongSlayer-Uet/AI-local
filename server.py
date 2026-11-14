import os
import tempfile
import time
import uuid
from datetime import datetime, timezone

import httpx
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from vieneu import Vieneu

# ============================================================
# CẤU HÌNH - sửa ở đây
# ============================================================
SERVER_HOST = "100.80.26.125"   # IP server lắng nghe, sửa nếu cần (vd: tailscale IP)
SERVER_PORT = 8000

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3:8b"
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Do not use thinking mode. "
    "Answer directly and concisely in 2-3 sentences. Avoid unnecessary explanations."
)

WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
WHISPER_LANGUAGE = "vietnamese"  # đổi hoặc để None nếu muốn Whisper tự detect ngôn ngữ

TTS_VOICE = "Trúc Ly"
AUDIO_OUTPUT_DIR = "audio_output"
AUDIO_OUTPUT_PATH = os.path.join(AUDIO_OUTPUT_DIR, "audio.wav")  # ghi đè mỗi lần

os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ============================================================
# LOAD MODEL NẶNG - 1 lần duy nhất lúc khởi động, không load lại mỗi request
# ============================================================
print("Loading Whisper (STT)...")
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_MODEL_ID, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
)
whisper_model.to(device)
whisper_model.generation_config.forced_decoder_ids = None
whisper_processor = AutoProcessor.from_pretrained(
    WHISPER_MODEL_ID, clean_up_tokenization_spaces=False
)
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    dtype=dtype,
    device=device,
    generate_kwargs={"language": WHISPER_LANGUAGE, "task": "transcribe"},
)

print("Loading VieNeu-TTS v3 Turbo...")
_tts_t0 = time.time()
tts = Vieneu(mode="v3turbo")
print(f"VieNeu-TTS loaded in {time.time() - _tts_t0:.2f}s")

app = FastAPI(title="AI Local Server", version="2.0.0")


class TextRequest(BaseModel):
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


def _transcribe_sync(filepath: str) -> str:
    return asr_pipe(filepath)["text"]


async def transcribe_audio(filepath: str) -> str:
    """STT, CPU/GPU-bound -> chạy trong threadpool để không block event loop."""
    return await run_in_threadpool(_transcribe_sync, filepath)


def _synthesize_speech_sync(text: str) -> str:
    audio = tts.infer(text, voice=TTS_VOICE)
    tts.save(audio, AUDIO_OUTPUT_PATH)
    return AUDIO_OUTPUT_PATH


async def synthesize_speech(text: str) -> str:
    """TTS, nặng & đồng bộ -> chạy trong threadpool để không block event loop."""
    return await run_in_threadpool(_synthesize_speech_sync, text)


@app.get("/health")
async def health():
    return {"status": "online", "timestamp": now()}


@app.post("/text")
async def text(request: TextRequest):
    """Nhận text, trả về text (chỉ qua LLM, không qua STT/TTS)."""
    reply = await ask_ollama(request.message)
    return {"response": reply, "timestamp": now()}


@app.post("/audio")
async def audio(file: UploadFile = File(...)):
    """Nhận audio (.wav), chạy full pipeline STT -> LLM -> TTS, trả về audio (.wav)."""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Only .wav files are supported")

    audio_bytes = await file.read()
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    try:
        transcript = await transcribe_audio(tmp_path)
    finally:
        os.remove(tmp_path)

    reply = await ask_ollama(transcript)
    audio_path = await synthesize_speech(reply)

    return FileResponse(audio_path, media_type="audio/wav", filename="response.wav")


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
