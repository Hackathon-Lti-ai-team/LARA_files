import json
import requests
import time
from datetime import datetime
from pathlib import Path
from tts import speak

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"
LOG_FILE = Path("conversation_log_project.json")

# ---------------- JSON LOG ----------------
def load_log():
    if LOG_FILE.exists():
        data = json.loads(LOG_FILE.read_text())
        data.setdefault("conversations", [])
        data.setdefault("model", OLLAMA_MODEL)
        data.setdefault("session_start", datetime.now().isoformat())
        return data

    return {
        "session_start": datetime.now().isoformat(),
        "model": OLLAMA_MODEL,
        "conversations": []
    }

def save_log(data):
    LOG_FILE.write_text(json.dumps(data, indent=2))


conversation_log = load_log()

# ---------------- WARM OLLAMA ----------------
def warm_ollama():
    try:
        requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": "Hello"},
            timeout=5
        )
    except Exception:
        pass


# ---------------- ASK OLLAMA ----------------
def ask_ollama(user_text: str, session):
    """
    Stream LLM response, speak in chunks, and log everything.
    """
    turn_id = len(conversation_log["conversations"]) + 1
    question_time = time.time()

    system_prompt = (
        "You are a real-time voice assistant.\n"
        "Speak naturally, clearly, and concisely.\n"
        "Keep it very short.\n"
        "Avoid long monologues.\n"
        "Never invent personal details.\n"
        f"{session.llm_identity_context()}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "stream": True,
        "options": {
            "temperature": 0.3,
            "num_predict": 40
        },
        "prompt": (
            f"System:\n{system_prompt}\n\n"
            f"User:\n{user_text}\n\n"
            f"Assistant:"
        )
    }

    full_response = ""
    speak_buffer = ""
    first_token_time = None

    try:
        with requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=20
        ) as r:

            r.raise_for_status()

            for line in r.iter_lines():
                if not line:
                    continue

                try:
                    token = json.loads(line.decode()).get("response", "")
                except Exception:
                    continue

                if not token:
                    continue

                if first_token_time is None:
                    first_token_time = time.time()

                full_response += token
                speak_buffer += token

                # Speak in chunks of 8 words
                words = speak_buffer.split()
                while len(words) >= 8:
                    chunk = " ".join(words[:8])
                    speak(chunk)
                    words = words[8:]
                speak_buffer = " ".join(words)  # remaining words

            # Speak any leftover words at the end
            if speak_buffer.strip():
                speak(speak_buffer.strip())

    except Exception as e:
        print("❌ Ollama error:", e)
        speak("Sorry, I had trouble responding.")
        full_response = ""

    tts_end_time = time.time()

    # ---------------- LOG ----------------
    conversation_log["conversations"].append({
        "turn": turn_id,
        "user": session.user_name if session.allow_name_usage else "Unknown",
        "user_text": user_text,
        "assistant": full_response.strip(),
        "timing": {
            "question_time": datetime.fromtimestamp(question_time).isoformat(),
            "tts_start_time": datetime.fromtimestamp(first_token_time).isoformat() if first_token_time else None,
            "tts_end_time": datetime.fromtimestamp(tts_end_time).isoformat(),
            "llm_latency_sec": round(first_token_time - question_time, 3) if first_token_time else None,
            "total_response_time_sec": round(tts_end_time - question_time, 3)
        }
    })

    save_log(conversation_log)

    return full_response.strip()
