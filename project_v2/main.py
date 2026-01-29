# main.py
# Orchestrator: wake-word → face → conversation pipeline

# main.py
# Orchestrator: wake → face → conversation

import time
from datetime import datetime
import json
from pathlib import Path

from session import SessionState
from camera import Camera
from face import detect_face, register_face
from audio import wait_for_wake_word, stop_mic
from stt import transcribe
from llm import ask_ollama, warm_ollama
from tts import speak
from emotion import load_sentiment_model, classify_emotion


# ---------------- CONFIG ----------------
LOG_FILE = Path("conversation_log_project.json")
FACE_CAPTURE_SECONDS = 5
IDLE_TIMEOUT = 15
POST_TTS_IDLE = 10

# ---------------- LOAD / SAVE LOG ----------------
def load_log():
    if LOG_FILE.exists():
        return json.loads(LOG_FILE.read_text())
    return {
        "session_start": datetime.now().isoformat(),
        "model": "phi3:mini",
        "conversations": []
    }

def save_log(data):
    LOG_FILE.write_text(json.dumps(data, indent=2))

conversation_log = load_log()
TURN_COUNTER = len(conversation_log.get("conversations", []))

# ---------------- STATES ----------------
IDLE = "IDLE"
FACE = "FACE"
CONVERSATION = "CONVERSATION"
SHUTDOWN = "SHUTDOWN"

# ---------------- MAIN ----------------
def main():
    global TURN_COUNTER

    print("\n🤖 LARA starting...")

    session = SessionState()
    warm_ollama()

    state = IDLE
    last_voice_time = time.time()
    last_tts_time = None
    last_user_text = None

    while state != SHUTDOWN:

        # ---------------- IDLE (VAD ONLY) ----------------
        if state == IDLE:
            
            if wait_for_wake_word(timeout=2.0):
                print("\n🤖 LARA starting...")
                print("🟢 Wake word detected")
                last_voice_time = time.time()

                # Load emotion model ONLY now
                load_sentiment_model()

                # Release mic before camera
                stop_mic()
                state = FACE
                continue


            if time.time() - last_voice_time > IDLE_TIMEOUT:
                print("⏱️ No voice detected. Shutting down.")
                state = SHUTDOWN
                continue

        # ---------------- FACE ----------------
        if state == FACE:
            print("📸 Capturing face...")
            cam = Camera()
            start = time.time()
            best_frame = None

            while time.time() - start < FACE_CAPTURE_SECONDS:
                ret, frame = cam.read()
                if ret:
                    best_frame = frame

            cam.release()

            name = detect_face(best_frame) if best_frame is not None else None

            # 🔒 Mic must stay OFF during TTS
            if name:
                session.set_known_user(name)
                speak(f"Hello {name}. You can speak now.")
            else:
                speak("I do not recognize you. Please type your name.")
                typed_name = input("Enter your name (or Enter to skip): ").strip()
                if typed_name and register_face(best_frame, typed_name):
                    session.set_known_user(typed_name)
                    speak(f"Thank you {typed_name}. Let's begin.")
                else:
                    session.set_anonymous()
                    speak("Alright. Continuing anonymously.")

            last_tts_time = time.time()
            state = CONVERSATION
            continue

        # ---------------- CONVERSATION ----------------
        if state == CONVERSATION:
            print("🎤 Listening...")
            audio = transcribe(blocking=True)
            current_time = time.time()

            if audio:
                user_text = audio.strip()
                if not user_text or user_text == last_user_text:
                    continue

                # 🎭 Emotion from STT output
                emotion = classify_emotion(user_text)

                if not user_text or user_text == last_user_text:
                    continue

                last_user_text = user_text
                TURN_COUNTER += 1
                question_time = current_time

                llm_text = ask_ollama(user_text, session)
                last_tts_time = time.time()
                last_voice_time = current_time

                conversation_log["conversations"].append({
                "turn": TURN_COUNTER,
                "user": session.user_name or "Anonymous",
                "user_text": user_text,
                "emotion": emotion,
                "assistant": llm_text,
                "timing": {
                    "question_time": datetime.fromtimestamp(question_time).isoformat(),
                    "tts_end_time": datetime.fromtimestamp(last_tts_time).isoformat(),
                    "total_response_time_sec": round(last_tts_time - question_time, 3)
                }
            })

                save_log(conversation_log)

            else:
                if last_tts_time and (current_time - last_tts_time > POST_TTS_IDLE):
                    print("⏱️ Conversation idle. Ending session.")
                    state = SHUTDOWN

    stop_mic()
    print("🛑 LARA stopped cleanly")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()


"""
import time
from datetime import datetime
import json
from pathlib import Path

from session import SessionState
from camera import Camera
from face import detect_face, register_face
from audio import vad_listen, stop_vad
from stt import transcribe
from llm import ask_ollama, warm_ollama
from tts import speak

# ---------------- CONFIG ----------------
LOG_FILE = Path("conversation_log_project.json")
FACE_CAPTURE_SECONDS = 5
IDLE_TIMEOUT = 15       # seconds for initial idle
POST_TTS_IDLE = 10      # seconds after TTS to shutdown

# ---------------- LOAD / SAVE LOG ----------------
def load_log():
    if LOG_FILE.exists():
        return json.loads(LOG_FILE.read_text())
    return {
        "session_start": datetime.now().isoformat(),
        "model": "phi3:mini",
        "conversations": []
    }

def save_log(data):
    LOG_FILE.write_text(json.dumps(data, indent=2))

conversation_log = load_log()
TURN_COUNTER = len(conversation_log.get("conversations", []))

# ---------------- STATES ----------------
IDLE = "IDLE"
FACE = "FACE"
CONVERSATION = "CONVERSATION"
SHUTDOWN = "SHUTDOWN"

# ---------------- MAIN ----------------
def main():
    global TURN_COUNTER

    print("\n🤖 LARA starting... Say 'anything' to begin")

    session = SessionState()
    warm_ollama()

    state = IDLE
    last_voice_time = time.time()
    last_tts_time = None       # track when last TTS finished
    last_user_text = None      # track last user input to avoid repeated TTS

    while state != SHUTDOWN:

        # ---------------- IDLE ----------------
        if state == IDLE:
            text = vad_listen(timeout=0.5)
            stop_vad()
            state = FACE
            if text:
                last_voice_time = time.time()
                print("👂 Voice detected")
                
                continue

            if time.time() - last_voice_time > IDLE_TIMEOUT:
                print("⏱️ No voice detected. Shutting down.")
                state = SHUTDOWN
                continue

        # ---------------- FACE CAPTURE ----------------
        # ---------------- FACE CAPTURE ----------------
        if state == FACE:
            print("📸 Capturing face...")
            cam = Camera()
            start = time.time()
            best_frame = None

            while time.time() - start < FACE_CAPTURE_SECONDS:
                ret, frame = cam.read()
                if ret:
                    best_frame = frame

            cam.release()

            name = detect_face(best_frame) if best_frame is not None else None

            if name:
                session.set_known_user(name)
                # Direct TTS for greeting
                speak(f"Hello {name}. You can speak now.")
            else:
                speak("I do not recognize you. Please type your name if you want to register.")
                typed_name = input("Enter your name (or press Enter to skip): ").strip()
                if typed_name and register_face(best_frame, typed_name):
                    session.set_known_user(typed_name)
                    speak(f"Thank you {typed_name}. Let's begin.")
                else:
                    session.set_anonymous()
                    speak("Alright. Continuing anonymously.")

            state = CONVERSATION
            continue


        # ---------------- CONVERSATION LOOP ----------------
        if state == CONVERSATION:
            print("🎤 Listening...")
            audio = transcribe(blocking=True)
            current_time = time.time()

            if audio:
                user_text = audio.strip()
                # skip repeated or empty text
                if not user_text or user_text == last_user_text:
                    continue

                last_user_text = user_text  # store current text

                TURN_COUNTER += 1
                question_time = current_time

                # Ask LLM and speak inside ask_ollama (do NOT call speak again)
                llm_text = ask_ollama(user_text, session)
                last_tts_time = time.time()      # mark TTS completion
                last_voice_time = current_time   # mark user voice

                # Save conversation log
                conversation_log["conversations"].append({
                    "turn": TURN_COUNTER,
                    "user": session.user_name or "Anonymous",
                    "user_text": user_text,
                    "assistant": llm_text,
                    "timing": {
                        "question_time": datetime.fromtimestamp(question_time).isoformat(),
                        "tts_end_time": datetime.fromtimestamp(last_tts_time).isoformat(),
                        "total_response_time_sec": round(last_tts_time - question_time, 3)
                    }
                })
                save_log(conversation_log)

            else:
                # No speech detected, check post-TTS idle
                if last_tts_time and (current_time - last_tts_time > POST_TTS_IDLE):
                    print(f"⏱️ No voice detected for {POST_TTS_IDLE}s after TTS. Ending session.")
                    state = SHUTDOWN

    print("🛑 LARA stopped cleanly")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()

"""
"""

import time
from datetime import datetime
import json
from pathlib import Path

from session import SessionState
from camera import Camera
from face import detect_face, register_face
from audio import vad_listen, stop_vad
from stt import transcribe
from llm import ask_ollama, warm_ollama
from tts import speak

# ---------------- CONFIG ----------------
LOG_FILE = Path("conversation_log_project.json")
FACE_CAPTURE_SECONDS = 5
IDLE_TIMEOUT = 15  # seconds for initial idle
POST_TTS_IDLE = 10  # seconds after TTS to shutdown

# ---------------- LOAD / SAVE LOG ----------------
def load_log():
    if LOG_FILE.exists():
        return json.loads(LOG_FILE.read_text())
    return {
        "session_start": datetime.now().isoformat(),
        "model": "phi3:mini",
        "conversations": []
    }

def save_log(data):
    LOG_FILE.write_text(json.dumps(data, indent=2))

conversation_log = load_log()
TURN_COUNTER = len(conversation_log.get("conversations", []))

# ---------------- STATES ----------------
IDLE = "IDLE"
FACE = "FACE"
CONVERSATION = "CONVERSATION"
SHUTDOWN = "SHUTDOWN"

# ---------------- MAIN ----------------
def main():
    global TURN_COUNTER

    print("\n🤖 LARA starting... Say 'anything' to begin")

    session = SessionState()
    warm_ollama()

    state = IDLE
    last_voice_time = time.time()
    last_tts_time = None          # track when last TTS finished
    last_user_text = None         # track last user input to avoid repeated TTS

    while state != SHUTDOWN:

        # ---------------- IDLE ----------------
        if state == IDLE:
            text = vad_listen(timeout=0.5)
            if text:
                last_voice_time = time.time()
                print("👂 Voice detected")
                stop_vad()
                state = FACE
                continue

            if time.time() - last_voice_time > IDLE_TIMEOUT:
                print("⏱️ No voice detected. Shutting down.")
                state = SHUTDOWN
                continue

        # ---------------- FACE CAPTURE ----------------
        if state == FACE:
            print("📸 Capturing face...")
            cam = Camera()
            start = time.time()
            best_frame = None

            while time.time() - start < FACE_CAPTURE_SECONDS:
                ret, frame = cam.read()
                if ret:
                    best_frame = frame

            cam.release()

            name = detect_face(best_frame) if best_frame is not None else None

            if name:
                session.set_known_user(name)
                speak(f"Hello {name}. You can speak now.")
            else:
                speak("I do not recognize you. Please type your name if you want to register.")
                typed_name = input("Enter your name (or press Enter to skip): ").strip()
                if typed_name and register_face(best_frame, typed_name):
                    session.set_known_user(typed_name)
                    speak(f"Thank you {typed_name}. Let's begin.")
                else:
                    session.set_anonymous()
                    speak("Alright. Continuing anonymously.")

            state = CONVERSATION
            continue

        # ---------------- CONVERSATION LOOP ----------------
        if state == CONVERSATION:
            print("🎤 Listening...")
            audio = transcribe(blocking=True)
            current_time = time.time()

            if audio:
                user_text = audio.strip()
                # skip repeated or empty text
                if not user_text or user_text == last_user_text:
                    continue

                last_user_text = user_text  # store current text

                TURN_COUNTER += 1
                question_time = current_time

                llm_text = ask_ollama(user_text, session)
                speak(llm_text)
                last_tts_time = time.time()      # mark TTS completion
                last_voice_time = current_time   # mark user voice

                # Save conversation log
                conversation_log["conversations"].append({
                    "turn": TURN_COUNTER,
                    "user": session.user_name or "Anonymous",
                    "user_text": user_text,
                    "assistant": llm_text,
                    "timing": {
                        "question_time": datetime.fromtimestamp(question_time).isoformat(),
                        "tts_end_time": datetime.fromtimestamp(last_tts_time).isoformat(),
                        "total_response_time_sec": round(last_tts_time - question_time, 3)
                    }
                })
                save_log(conversation_log)

            else:
                # No speech detected, check post-TTS idle
                if last_tts_time and (current_time - last_tts_time > POST_TTS_IDLE):
                    print(f"⏱️ No voice detected for {POST_TTS_IDLE}s after TTS. Ending session.")
                    state = SHUTDOWN

    print("🛑 LARA stopped cleanly")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()
"""

"""

# main.py
# Orchestrator: wake-word → face → conversation pipeline

import time
from datetime import datetime
import json
from pathlib import Path

from session import SessionState
from camera import Camera
from face import detect_face, register_face
from audio import vad_listen, stop_vad
from stt import transcribe
from llm import ask_ollama, warm_ollama
from tts import speak

# ---------------- CONFIG ----------------
LOG_FILE = Path("conversation_log_project.json")
WAKE_WORD = "machine"  # wake word
FACE_CAPTURE_SECONDS = 5
IDLE_TIMEOUT = 15  # seconds for initial idle
POST_TTS_IDLE = 10  # seconds of silence after TTS to end session

# ---------------- LOAD / SAVE LOG ----------------
def load_log():
    if LOG_FILE.exists():
        return json.loads(LOG_FILE.read_text())
    return {
        "session_start": datetime.now().isoformat(),
        "model": "phi3:mini",
        "conversations": []
    }

def save_log(data):
    LOG_FILE.write_text(json.dumps(data, indent=2))

conversation_log = load_log()
TURN_COUNTER = len(conversation_log.get("conversations", []))

# ---------------- STATES ----------------
IDLE = "IDLE"
FACE = "FACE"
CONVERSATION = "CONVERSATION"
SHUTDOWN = "SHUTDOWN"

# ---------------- MAIN ----------------
def main():
    global TURN_COUNTER

    print("\n🤖 LARA starting... Say 'Hello LARA' to begin")

    session = SessionState()
    warm_ollama()

    state = IDLE
    last_voice_time = time.time()
    last_tts_time = None  # track when last TTS finished
    last_user_text = None  # avoid repeated TTS

    while state != SHUTDOWN:

        # ---------------- IDLE / WAKE WORD ----------------
        if state == IDLE:
            text = vad_listen(timeout=0.5)
            if text:
                last_voice_time = time.time()
                # robust wake word detection: check all words in WAKE_WORD
                wake_words = WAKE_WORD.lower().split()
                text_words = text.lower().split()
                if all(word in text_words for word in wake_words):
                    print("👂 Wake word detected")
                    stop_vad()
                    state = FACE
                    continue
                else:
                    # detected speech but not wake word
                    continue

            # shutdown if no wake word detected for IDLE_TIMEOUT
            if time.time() - last_voice_time > IDLE_TIMEOUT:
                print("⏱️ No wake word detected. Shutting down.")
                state = SHUTDOWN
                continue

        # ---------------- FACE CAPTURE ----------------
        if state == FACE:
            print("📸 Capturing face...")
            cam = Camera()
            start = time.time()
            best_frame = None

            while time.time() - start < FACE_CAPTURE_SECONDS:
                ret, frame = cam.read()
                if ret:
                    best_frame = frame

            cam.release()

            name = detect_face(best_frame) if best_frame is not None else None

            if name:
                session.set_known_user(name)
                speak(f"Hello {name}. You can speak now.")
            else:
                speak("I do not recognize you. Please type your name if you want to register.")
                typed_name = input("Enter your name (or press Enter to skip): ").strip()
                if typed_name and register_face(best_frame, typed_name):
                    session.set_known_user(typed_name)
                    speak(f"Thank you {typed_name}. Let's begin.")
                else:
                    session.set_anonymous()
                    speak("Alright. Continuing anonymously.")

            state = CONVERSATION
            continue

        # ---------------- CONVERSATION LOOP ----------------
        if state == CONVERSATION:
            print("🎤 Listening...")
            audio = transcribe(blocking=True)
            current_time = time.time()

            if audio:
                user_text = audio.strip()
                if not user_text or user_text == last_user_text:
                    continue  # skip repeated or empty text

                last_user_text = user_text  # store current text

                TURN_COUNTER += 1
                question_time = current_time

                llm_text = ask_ollama(user_text, session)
                speak(llm_text)
                last_tts_time = time.time()      # mark TTS completion
                last_voice_time = current_time   # mark user voice

                # Save conversation log
                conversation_log["conversations"].append({
                    "turn": TURN_COUNTER,
                    "user": session.user_name or "Anonymous",
                    "user_text": user_text,
                    "assistant": llm_text,
                    "timing": {
                        "question_time": datetime.fromtimestamp(question_time).isoformat(),
                        "tts_end_time": datetime.fromtimestamp(last_tts_time).isoformat(),
                        "total_response_time_sec": round(last_tts_time - question_time, 3)
                    }
                })
                save_log(conversation_log)

            else:
                # No speech detected, check post-TTS idle
                if last_tts_time and (current_time - last_tts_time > POST_TTS_IDLE):
                    print(f"⏱️ No voice detected for {POST_TTS_IDLE}s after TTS. Ending session.")
                    state = SHUTDOWN

    print("🛑 LARA stopped cleanly")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()
"""