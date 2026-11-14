#!/usr/bin/env python3
"""
Entry point cho client tren Raspberry Pi.
Hien tai chi bat tinh nang VAD capture (nghe mic, cat cau, goi AI-server).
Sau nay se ghep them cac tinh nang khac (TTS Piper, OLED, Gemini Live...) vao day.
"""
from vad.vad_capture import run_mic

if __name__ == "__main__":
    run_mic()