#!/usr/bin/env python3
"""
Silero VAD (ONNX) cho Raspberry Pi 4.
Sau khi cat duoc 1 cau noi, tu dong gui file wav len AI-server (/api/v1/audio)
de lay transcript + response, roi in ra man hinh.

Cach dung:
    python3 vad_capture.py              # nghe mic truc tiep, Ctrl+C de thoat
    python3 vad_capture.py test.wav     # phan tich file wav (16kHz, mono, S16_LE)

Cai dat:
    pip install onnxruntime numpy requests
"""
import os
import sys
import time
import wave
import subprocess
from collections import deque

import numpy as np
import onnxruntime as ort
import requests

# ---------- Cau hinh ----------
# Luon tinh theo vi tri file nay (khong dung os.getcwd()), vi main.py o thu muc cha
# se import module nay, luc do cwd la thu muc cha chu khong phai thu muc vad/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "silero_vad.onnx")
ALSA_DEVICE = "plughw:4,0"
SR = 16000
CHUNK = 512                 # Silero yeu cau 512 mau @16kHz (32 ms)
CHUNK_MS = CHUNK * 1000 // SR

THRESHOLD = 0.5             # prob >= nguong nay -> co tieng noi
NEG_THRESHOLD = THRESHOLD - 0.15
MIN_SILENCE_MS = 500        # im lang bao lau thi coi la het cau
MIN_SPEECH_MS = 250         # doan noi ngan hon thi bo (nhieu)
PRE_ROLL_MS = 300           # giu lai truoc luc bat dau noi de khong mat dau cau
SAVE_SEGMENTS = True        # luu cau thanh file wav

OUTPUT_DIR = os.path.join(BASE_DIR, "recordings")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "segment.wav")

SERVER_URL = "http://100.74.177.102:8000/api/v1/audio"
REQUEST_TIMEOUT = 60        # giay, chu yeu cho whisper + llm xu ly xong ben server


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


def save_wav(path, audio):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(audio.tobytes())


def send_to_server(path):
    """Gui file wav len AI-server. Tra ve dict {"transcript", "response", ...} hoac None neu loi."""
    try:
        with open(path, "rb") as f:
            r = requests.post(SERVER_URL, files={"file": f}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"            Loi goi AI-server: {e}")
        return None


def handle(event, audio, t):
    if event == "start":
        print(f"[{t:7.2f}s] >>> bat dau noi")
    elif event == "end":
        dur = len(audio) / SR
        print(f"[{t:7.2f}s] <<< het cau ({dur:.2f}s)")
        if SAVE_SEGMENTS:
            save_wav(OUTPUT_PATH, audio)
            print(f"            ghi de {OUTPUT_PATH}")

        result = send_to_server(OUTPUT_PATH)
        if result:
            print(f"            STT     : {result.get('transcript', '').strip()}")
            print(f"            Tra loi : {result.get('response', '').strip()}")


def run_file(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, \
            "Can file 16kHz, mono, S16_LE"
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    seg = SpeechSegmenter(SileroVAD(MODEL_PATH))
    t0 = time.time()
    for i in range(0, len(data) - CHUNK + 1, CHUNK):
        _, event, audio = seg.feed(data[i:i + CHUNK])
        handle(event, audio, i / SR)
    # file ket thuc khi dang noi do -> luu not
    if seg.in_speech and seg.buf:
        handle("end", np.concatenate(seg.buf), len(data) / SR)

    elapsed = time.time() - t0
    print(f"Xu ly {len(data)/SR:.1f}s audio trong {elapsed:.2f}s")


def run_mic():
    cmd = ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE", "-r", str(SR),
           "-c", "1", "-t", "raw", "-q"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    seg = SpeechSegmenter(SileroVAD(MODEL_PATH))
    n = 0
    print("Dang nghe... (Ctrl+C de thoat)")
    try:
        while True:
            raw = proc.stdout.read(CHUNK * 2)  # 2 byte / mau
            if len(raw) < CHUNK * 2:
                print("arecord dong. Kiem tra lai thiet bi ALSA.")
                break
            chunk = np.frombuffer(raw, dtype=np.int16)
            prob, event, audio = seg.feed(chunk)
            handle(event, audio, n * CHUNK / SR)
            n += 1
    except KeyboardInterrupt:
        print("\nThoat.")
    finally:
        proc.terminate()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    else:
        run_mic()