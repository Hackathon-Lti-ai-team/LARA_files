# tts.py
# Text-to-Speech (blocking, mic-safe)

import subprocess
import threading

# Lock to prevent overlapping speech
_tts_lock = threading.Lock()


def speak(text: str):
    """
    Blocking TTS.
    While this runs, mic/VAD must be OFF (enforced by main.py).
    """
    if not text or not text.strip():
        return

    with _tts_lock:
        try:
            subprocess.run(
                [
                    "espeak-ng",
                    "-v", "en+f3",
                    "-s", "155",
                    "-p", "55",
                    "-a", "180",
                    "-g", "0",
                    "-k", "10",
                    text
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        except Exception as e:
            print("❌ TTS error:", e)
