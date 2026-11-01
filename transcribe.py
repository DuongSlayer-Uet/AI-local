from faster_whisper import WhisperModel
import time

print("Loading Whisper Large-v3...")

start = time.time()

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"
)

print(f"Model loaded in {time.time() - start:.2f}s")

start = time.time()

segments, info = model.transcribe(
    "audio.wav",
    language="vi",
    beam_size=5,
    vad_filter=True
)

segments = list(segments)

print(f"\nLanguage: {info.language}")

for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

print(f"\nInference time: {time.time() - start:.2f}s")
