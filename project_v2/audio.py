# audio.py
# Handles ONLY VAD-based listening and mic ownership
"""
import queue
import time
import numpy as np
import sounddevice as sd
import webrtcvad

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

_vad = webrtcvad.Vad(2)
_vad_stream = None
_vad_running = False
_audio_q = queue.Queue()


def _vad_callback(indata, frames, time_info, status):
    if not _vad_running:
        return
    _audio_q.put(bytes(indata))


def vad_listen(timeout=0.5):

    global _vad_stream, _vad_running

    if _vad_stream is None:
        _vad_running = True
        _vad_stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=_vad_callback
        )
        _vad_stream.start()

    start = time.time()
    voiced_frames = []

    while time.time() - start < timeout:
        try:
            frame = _audio_q.get(timeout=timeout)
        except queue.Empty:
            continue

        if _vad.is_speech(frame, SAMPLE_RATE):
            voiced_frames.append(frame)

    if voiced_frames:
        # Return dummy marker; actual wake-word STT happens elsewhere
        return "speech"

    return None


def stop_vad():
    global _vad_stream, _vad_running

    _vad_running = False
    if _vad_stream:
        _vad_stream.stop()
        _vad_stream.close()
        _vad_stream = None

    while not _audio_q.empty():
        _audio_q.get()
"""

# audio.py
# audio.py
# Unified mic ownership: VAD + Wake-word (Vosk)
# SAFE for camera + TTS + Pi

import queue
import time
import json
import sounddevice as sd
import webrtcvad
from vosk import Model, KaldiRecognizer

# ---------------- CONFIG ----------------

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

VOSK_CHUNK = 4000
WAKE_WORDS = [
    "hi lara",
    "hi laura",
    "hi laara",
    "hi laraa",
    "hi lora"]   # change later
MODEL_PATH = "vosk-model-small-en-us-0.15"

# ---------------- STATE ----------------

_vad = webrtcvad.Vad(2)

_model = Model(MODEL_PATH)
_rec = KaldiRecognizer(_model, SAMPLE_RATE)

_stream = None
_running = False

# Bounded queue = NO camera lag
_audio_q = queue.Queue(maxsize=30)

# ---------------- AUDIO CALLBACK ----------------

def _audio_callback(indata, frames, time_info, status):
    if not _running:
        return

    try:
        _audio_q.put_nowait(bytes(indata))
    except queue.Full:
        # Drop frames if overloaded (critical for FPS)
        pass

# ---------------- MIC CONTROL ----------------

def start_mic():
    global _stream, _running

    if _stream is not None:
        return

    _running = True
    _stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=_audio_callback
    )
    _stream.start()


def stop_mic():
    global _stream, _running

    _running = False

    if _stream:
        _stream.stop()
        _stream.close()
        _stream = None

    # Flush audio
    while not _audio_q.empty():
        try:
            _audio_q.get_nowait()
        except queue.Empty:
            break

# ---------------- VAD ----------------

def vad_listen(timeout=0.5):
    """
    Returns True if any speech is detected.
    """
    start_mic()
    start = time.time()

    while time.time() - start < timeout:
        try:
            frame = _audio_q.get(timeout=0.05)
        except queue.Empty:
            continue

        if _vad.is_speech(frame, SAMPLE_RATE):
            return True

    return False

# ---------------- WAKE WORD ----------------

def wait_for_wake_word(timeout=2.0):
    """
    Returns True if wake word detected.
    """
    start_mic()
    start = time.time()

    buffer = b""

    while time.time() - start < timeout:
        try:
            frame = _audio_q.get(timeout=0.05)
        except queue.Empty:
            continue

        buffer += frame

        if len(buffer) >= VOSK_CHUNK:
            if _rec.AcceptWaveform(buffer):
                result = json.loads(_rec.Result())
                text = result.get("text", "").lower()
                print("🗣️ Wake recognizer:", text)

                if any(w in text for w in WAKE_WORDS):
                    return True


            buffer = b""

    return False
