#!/usr/bin/env python3
"""
Client cho Raspberry Pi: Silero VAD (ONNX) cat cau tu mic -> gui AI-server
(/audio) -> nhan ve audio tra loi -> phat qua pw-play. Khong luu file audio
nao ca (gui/nhan deu bang bytes trong RAM).

Trong luc dang phat am thanh tra loi, VAD tam dung (bo qua mic), phat xong
thi nghe tiep.

Cach dung:
    python3 client.py        # Ctrl+C de thoat

Cai dat:
    pip install onnxruntime numpy requests
"""
import io
import os
import queue
import subprocess
import threading
import wave
from collections import deque

import numpy as np
import onnxruntime as ort
import requests

# ============================================================
# CAU HINH - sua o day
# ============================================================
SERVER_URL = "http://100.82.247.92:8000/audio"  # sua IP:port server o day
REQUEST_TIMEOUT = 60  # giay, cho whisper + llm + tts ben server xu ly xong

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "silero_vad.onnx")  # dat canh client.py
ALSA_DEVICE = "plughw:4,0"
SR = 16000
CHUNK = 512  # Silero yeu cau 512 mau @16kHz (32 ms)
CHUNK_MS = CHUNK * 1000 // SR

THRESHOLD = 0.5  # prob >= nguong nay -> co tieng noi
NEG_THRESHOLD = THRESHOLD - 0.15
MIN_SILENCE_MS = 500  # im lang bao lau thi coi la het cau
MIN_SPEECH_MS = 250  # doan noi ngan hon thi bo (nhieu)
PRE_ROLL_MS = 300  # giu lai truoc luc bat dau noi de khong mat dau cau


class SileroVAD:
    """Tu dong nhan dien model v5 (input 'state') hoac v4 (input 'h', 'c')."""

    def __init__(self, path, sr=SR):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(
            path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        names = [i.name for i in self.sess.get_inputs()]
        self.v5 = "state" in names
        self.sr = np.array(sr, dtype=np.int64)
        self.reset()

    def reset(self):
        if self.v5:
            self.state = np.zeros((2, 1, 128), dtype=np.float32)
            self.context = np.zeros((1, 64), dtype=np.float32)
        else:
            self.h = np.zeros((2, 1, 64), dtype=np.float32)
            self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, chunk_i16):
        x = (chunk_i16.astype(np.float32) / 32768.0)[None, :]
        if self.v5:
            x = np.concatenate([self.context, x], axis=1)  # them 64 mau context
            self.context = x[:, -64:]
            out, self.state = self.sess.run(
                None, {"input": x, "state": self.state, "sr": self.sr}
            )
        else:
            out, self.h, self.c = self.sess.run(
                None, {"input": x, "sr": self.sr, "h": self.h, "c": self.c}
            )
        return float(out[0][0])


class SpeechSegmenter:
    """May trang thai: chuyen xac suat tung chunk thanh cac cau noi."""

    def __init__(self, vad):
        self.vad = vad
        self.min_silence_chunks = MIN_SILENCE_MS // CHUNK_MS
        self.min_speech_chunks = MIN_SPEECH_MS // CHUNK_MS
        self.pre_roll = deque(maxlen=PRE_ROLL_MS // CHUNK_MS)
        self.reset()

    def reset(self):
        self.vad.reset()
        self.in_speech = False
        self.buf = []
        self.silence = 0
        self.speech_chunks = 0

    def feed(self, chunk):
        """Tra ve (prob, event, audio). event: None | 'start' | 'end'."""
        prob = self.vad(chunk)
        event, audio = None, None

        if not self.in_speech:
            self.pre_roll.append(chunk)
            if prob >= THRESHOLD:
                self.in_speech = True
                self.buf = list(self.pre_roll)
                self.silence = 0
                self.speech_chunks = 1
                event = "start"
        else:
            self.buf.append(chunk)
            if prob >= NEG_THRESHOLD:
                self.silence = 0
                self.speech_chunks += 1
            else:
                self.silence += 1
                if self.silence >= self.min_silence_chunks:
                    if self.speech_chunks >= self.min_speech_chunks:
                        event = "end"
                        audio = np.concatenate(self.buf)
                    self.in_speech = False
                    self.buf = []
                    self.pre_roll.clear()
                    self.vad.reset()  # reset state giua cac cau
        return prob, event, audio


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Dong goi audio (int16 mono 16kHz) thanh wav trong RAM, khong ghi file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


def send_to_server(wav_bytes: bytes):
    """Gui audio len AI-server (/audio). Tra ve audio tra loi (bytes) hoac None neu loi."""
    try:
        r = requests.post(
            SERVER_URL,
            files={"file": ("segment.wav", wav_bytes, "audio/wav")},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.content
    except requests.RequestException as e:
        print(f"            Loi goi AI-server: {e}")
        return None


def play_audio(wav_bytes: bytes):
    """Phat audio tra loi qua pw-play, truyen thang qua stdin, khong ghi file."""
    print("            >>> dang phat am thanh tra loi...")
    subprocess.run(["pw-play", "-"], input=wav_bytes)
    print("            <<< phat xong, tiep tuc nghe")


def mic_reader(proc, q):
    """Chay trong thread rieng: doc lien tuc tu arecord, day vao queue.
    Lam vay de du dang phat tra loi (main thread bi block), arecord van
    duoc doc het, khong bi nghen pipe."""
    while True:
        raw = proc.stdout.read(CHUNK * 2)  # 2 byte / mau
        if len(raw) < CHUNK * 2:
            q.put(None)
            return
        q.put(raw)


def handle(event, audio, t):
    """Xu ly 1 su kien VAD. Tra ve audio tra loi (bytes) neu co cau vua ket thuc."""
    if event == "start":
        print(f"[{t:7.2f}s] >>> bat dau noi")
    elif event == "end":
        dur = len(audio) / SR
        print(f"[{t:7.2f}s] <<< het cau ({dur:.2f}s)")
        return send_to_server(audio_to_wav_bytes(audio))
    return None


def run_mic():
    cmd = ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE", "-r", str(SR),
           "-c", "1", "-t", "raw", "-q"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    q = queue.Queue(maxsize=2000)
    threading.Thread(target=mic_reader, args=(proc, q), daemon=True).start()

    seg = SpeechSegmenter(SileroVAD(MODEL_PATH))
    playing = False
    n = 0
    print("Dang nghe... (Ctrl+C de thoat)")
    try:
        while True:
            raw = q.get()
            if raw is None:
                print("arecord dong. Kiem tra lai thiet bi ALSA.")
                break

            if playing:
                continue  # dang phat tra loi -> bo qua mic, khong feed VAD

            chunk = np.frombuffer(raw, dtype=np.int16)
            prob, event, audio = seg.feed(chunk)
            reply_audio = handle(event, audio, n * CHUNK / SR)

            if reply_audio:
                playing = True
                play_audio(reply_audio)
                seg.reset()  # bo qua am thanh/echo thu duoc trong luc dang phat
                playing = False

            n += 1
    except KeyboardInterrupt:
        print("\nThoat.")
    finally:
        proc.terminate()


if __name__ == "__main__":
    run_mic()

