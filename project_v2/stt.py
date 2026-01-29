# stt.py
# Speech-to-Text logic (wake-word + full utterance)

import queue
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

MODEL_PATH = "whisper_models/tiny.en"
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"

# Load whisper once
model = WhisperModel(
    MODEL_PATH,
    device="cpu",
    compute_type="int8",
    cpu_threads=4,
    num_workers=1
)


def _record_until_silence(max_silence=0.8, max_duration=6):
    """Record audio until silence is detected."""
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    frames = []
    silence_start = None
    start_time = time.time()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback
    ):
        while True:
            try:
                data = q.get(timeout=0.1)
            except queue.Empty:
                continue

            frames.append(data)
            energy = np.abs(data).mean()

            if energy < 0.003:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > max_silence:
                    break
            else:
                silence_start = None

            if time.time() - start_time > max_duration:
                break

    if not frames:
        return None

    return np.concatenate(frames, axis=0).flatten()


def transcribe(blocking=True):
    """
    Records speech (blocking), then runs Whisper STT.
    Used ONLY during active conversation or wake-word STT.
    """
    audio = _record_until_silence()
    if audio is None:
        return None

    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=1,
        best_of=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300)
    )

    text = " ".join(s.text.strip() for s in segments)
    return text.strip()