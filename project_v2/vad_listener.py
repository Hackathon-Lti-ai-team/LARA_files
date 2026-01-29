"""

import sounddevice as sd
import numpy as np
import webrtcvad
import time
import sys

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)

vad = webrtcvad.Vad(2)  # 0–3 (2 is balanced)

print("🎤 Mic active. Speak to test VAD...")

def audio_callback(indata, frames, time_info, status):
    pcm16 = (indata[:, 0] * 32768).astype(np.int16).tobytes()

    if vad.is_speech(pcm16, SAMPLE_RATE):
        print("🟢 VAD: SPEECH DETECTED")
        sys.stdout.flush()

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=FRAME_SIZE,
    dtype="float32",
    callback=audio_callback
):
    while True:
        time.sleep(0.1)
"""



import sounddevice as sd
import queue
import json
import cv2
import time
from vosk import Model, KaldiRecognizer

MODEL_PATH = "vosk-model-small-en-us-0.15"
RATE = 16000
CHUNK = 4000

q = queue.Queue()
cam = None

model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, RATE)

def open_camera():
    global cam
    if cam is None:
        cam = cv2.VideoCapture(1)
        print("📷 Camera ON (Wake word spotted)")

def audio_callback(indata, frames, time_info, status):
    q.put(bytes(indata))

with sd.RawInputStream(
    samplerate=RATE, blocksize=CHUNK,
    dtype="int16", channels=1,
    callback=audio_callback
):
    print("🎤 Say your wake word now…")

    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            text = json.loads(rec.Result())["text"]
            print("Recognized:", text)
            # <-- replace "hey robot" with your wake word
            if "fuck you bitch" in text.lower():
                open_camera()

        if cam:
            ret, frame = cam.read()
            if ret:
                cv2.imshow("Wake Camera", frame)
                if cv2.waitKey(1) == 27:
                    break
        time.sleep(0.01)

if cam:
    cam.release()
cv2.destroyAllWindows()



"""
import pyttsx3

engine = pyttsx3.init() # object creation

# RATE
rate = engine.getProperty('rate')   # getting details of current speaking rate
print (rate)                        # printing current voice rate
engine.setProperty('rate', 110)     # setting up new voice rate

# VOLUME
volume = engine.getProperty('volume')   # getting to know current volume level (min=0 and max=1)
print (volume)                          # printing current volume level
engine.setProperty('volume',1.0)        # setting up volume level  between 0 and 1

# VOICE
voices = engine.getProperty('voices')       # getting details of current voice
#engine.setProperty('voice', "english+f3")  # changing index, changes voices. o for male
engine.setProperty('voice', voices[1].id)   # changing index, changes voices. 1 for female

engine.say("Hello World!")
#engine.say('My current speaking rate is ' + str(rate))
engine.runAndWait()
engine.stop()
"""
