
from transformers import pipeline
import os

# -------- FORCE OFFLINE MODE --------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

sentiment_model = None  # lazy-loaded

CONFUSION_WORDS = [
    "confused", "don't understand", "unclear"
]

CURIOSITY_WORDS = [
    "what", "why", "how", "explain", "tell me", "can you", "who"
]

def load_sentiment_model():
    global sentiment_model
    if sentiment_model is None:
        print("[Emotion] Loading sentiment model...")
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            local_files_only=True
        )
        sentiment_model("hello")  # warm-up
        print("[Emotion] Model ready.")

def classify_emotion(text: str):
    if not text or not text.strip():
        return "NEUTRAL"

    clean_text = text.lower()
    sentiment = sentiment_model(text)[0]

    label = sentiment["label"]
    score = sentiment["score"]

    # PRIORITY 1
    if any(w in clean_text for w in CONFUSION_WORDS):
        return "CONFUSED"

    if "?" in text or any(w in clean_text for w in CURIOSITY_WORDS):
        return "CURIOUS"

    # PRIORITY 2
    if label == "POSITIVE" and score > 0.85:
        return "EXCITED"

    if label == "POSITIVE" and score > 0.6:
        return "HAPPY"

    # PRIORITY 3
    if label == "NEGATIVE" and score > 0.4:
        return "SAD"

    return "NEUTRAL"



"""
from transformers import pipeline
import os

# -------- FORCE OFFLINE MODE --------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# ------------------ LOAD MODEL ONCE ------------------

print("[Emotion] Loading sentiment model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    local_files_only=True
)
sentiment_model("hello")  # warm-up
print("[Emotion] Model ready.")

# ------------------ KEYWORDS ------------------

CONFUSION_WORDS = [
    "confused", "don't understand", "unclear"
]

CURIOSITY_WORDS = [
    "what", "why", "how", "explain", "tell me", "can you", "who"
]

# ------------------ CLASSIFIER ------------------

def classify_emotion(text: str):
    if not text or not text.strip():
        return "NEUTRAL"

    clean_text = text.lower()
    sentiment = sentiment_model(text)[0]

    label = sentiment["label"]
    score = sentiment["score"]

    # ---------------- PRIORITY 1 ----------------
    # CONFUSED
    if any(w in clean_text for w in CONFUSION_WORDS):
        return "CONFUSED"

    # CURIOUS
    if "?" in text or any(w in clean_text for w in CURIOSITY_WORDS):
        return "CURIOUS"

    # ---------------- PRIORITY 2 ----------------
    # EXCITED
    if label == "POSITIVE" and score > 0.85:
        return "EXCITED"

    # HAPPY
    if label == "POSITIVE" and score > 0.6:
        return "HAPPY"

    # ---------------- PRIORITY 3 ----------------
    # SAD
    if label == "NEGATIVE" and score > 0.4:
        return "SAD"

    return "NEUTRAL"

"""

"""
import os
import json
from collections import Counter
from transformers import pipeline
FRUSTRATION_WORDS = [
    "angry", "annoyed", "frustrated", "irritated",
    "not working", "problem", "error", "stuck"
]

SAD_WORDS = [
    "sad", "unhappy", "depressed", "lonely",
    "upset", "tired"
]

CONFUSION_WORDS = [
    "confused", "don't understand", "what",
    "why", "how", "explain"
]
print("Loading sentiment model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Model loaded.")
def get_emotion_state(user_text):
    text = user_text.lower()
    sentiment = sentiment_model(user_text)[0]

    label = sentiment["label"]
    score = sentiment["score"]

    if label == "POSITIVE" and score > 0.85 and "!" in user_text:
        emotion = "EXCITED"
    elif label == "POSITIVE" and score > 0.6:
        emotion = "HAPPY"
    elif label == "NEGATIVE" and score > 0.6 and any(w in text for w in FRUSTRATION_WORDS):
        emotion = "FRUSTRATED"
    elif label == "NEGATIVE" and any(w in text for w in SAD_WORDS):
        emotion = "SAD"
    elif any(w in text for w in CONFUSION_WORDS):
        emotion = "CONFUSED"
    else:
        emotion = "NEUTRAL"

    return emotion, label, round(score, 3)
def process_log_file(file_path):
    results = []

    with open(file_path, "r") as f:
        data = json.load(f)

    session_start = data.get("session_start")
    model = data.get("model")

    for turn in data.get("conversations", []):
        user_text = turn.get("user_text", "").strip()
        user = turn.get("user", "Unknown")
        turn_id = turn.get("turn")

        if not user_text:
            continue

        emotion, sentiment_label, sentiment_score = get_emotion_state(user_text)

        results.append({
            "session_start": session_start,
            "model": model,
            "turn": turn_id,
            "user": user,
            "user_text": user_text,
            "sentiment": sentiment_label,
            "sentiment_score": sentiment_score,
            "emotion": emotion
        })

    return results
LOG_DIR = "conversation_logs"
all_results = []

for file_name in os.listdir(LOG_DIR):
    if file_name.endswith(".json"):
        file_path = os.path.join(LOG_DIR, file_name)
        print(f"Processing {file_name}")
        all_results.extend(process_log_file(file_path))
for r in all_results:
    print("-" * 60)
    print(f"Session  : {r['session_start']}")
    print(f"Turn     : {r['turn']}")
    print(f"User     : {r['user']}")
    print(f"Text     : {r['user_text']}")
    print(f"Sentiment: {r['sentiment']} ({r['sentiment_score']})")
    print(f"Emotion  : {r['emotion']}")
emotion_counter = Counter(r["emotion"] for r in all_results)

print("\nEmotion Summary:")
for emotion, count in emotion_counter.items():
    print(f"{emotion:12}: {count}")

"""