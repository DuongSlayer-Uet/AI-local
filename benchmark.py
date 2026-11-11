import time
import requests
from faster_whisper import WhisperModel

# =====================================
# CONFIGURATION
# =====================================

AUDIO_FILE = "audio.wav"
MODEL_NAME = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# =====================================
# 1. WHISPER STT
# =====================================

print("\n========== WHISPER STT ==========")

start_total = time.perf_counter()

print("Loading Whisper Large-v3...")

start = time.perf_counter()

whisper = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"
)

whisper_load_time = time.perf_counter() - start

print(f"Model loaded: {whisper_load_time:.2f}s")

start = time.perf_counter()

segments, info = whisper.transcribe(
    AUDIO_FILE,
    language="vi",
    beam_size=5,
    vad_filter=True
)

segments = list(segments)

transcription = " ".join(
    segment.text.strip() for segment in segments
)

stt_time = time.perf_counter() - start

print(f"\nTranscription: {transcription}")
print(f"STT time: {stt_time:.2f}s")

# =====================================
# 2. QWEN3 LLM
# =====================================

print("\n========== QWEN3 LLM ==========")

prompt = f"""
Bạn là một trợ lý AI thông minh.
Hãy trả lời câu hỏi của người dùng bằng tiếng Việt,
Ngắn gọn dễ hiểu.
Người dùng nói:
{transcription}
"""

payload = {
    "model": MODEL_NAME,
    "prompt": prompt,
    "stream": False,
    "think": False
}

start = time.perf_counter()

response = requests.post(
    OLLAMA_URL,
    json=payload,
    timeout=300
)

response.raise_for_status()

result = response.json()

llm_time = time.perf_counter() - start

answer = result["response"]

print("\nAI Response:")
print(answer)

# =====================================
# 3. BENCHMARK
# =====================================

total_time = time.perf_counter() - start_total

print("\n========== BENCHMARK ==========")

print(f"Whisper loading : {whisper_load_time:.2f}s")
print(f"Whisper STT     : {stt_time:.2f}s")
print(f"Qwen3 LLM       : {llm_time:.2f}s")
print(f"Total pipeline  : {total_time:.2f}s")

print("\n========== LLM DETAILS ==========")

print(f"Prompt tokens   : {result.get('prompt_eval_count', 0)}")
print(f"Output tokens   : {result.get('eval_count', 0)}")

print(
    f"Prompt speed    : "
    f"{result.get('prompt_eval_rate', 0):.2f} tokens/s"
)

print(
    f"Generation speed: "
    f"{result.get('eval_rate', 0):.2f} tokens/s"
)

print("\nBenchmark completed!")
