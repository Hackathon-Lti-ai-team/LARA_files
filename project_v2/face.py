# face.py
# Face detection, recognition, and registration

import cv2
import os
import numpy as np
import face_recognition
import re

DB_PATH = "faces_db"
os.makedirs(DB_PATH, exist_ok=True)

# In-memory cache to avoid repeated disk IO
DB_CACHE = {
    "encodings": [],
    "names": [],
    "loaded": False
}


def sanitize_name(name: str) -> str:
    """Make name filesystem-safe."""
    return re.sub(r"[^\w\-]", "_", name.strip())


def load_database(force_reload=False):
    """Load all face encodings into memory."""
    if DB_CACHE["loaded"] and not force_reload:
        return DB_CACHE["encodings"], DB_CACHE["names"]

    encodings, names = [], []

    for person in os.listdir(DB_PATH):
        person_dir = os.path.join(DB_PATH, person)
        if not os.path.isdir(person_dir):
            continue

        for f in os.listdir(person_dir):
            try:
                enc = np.load(os.path.join(person_dir, f))
                encodings.append(enc)
                names.append(person)
            except Exception:
                continue

    DB_CACHE["encodings"] = encodings
    DB_CACHE["names"] = names
    DB_CACHE["loaded"] = True

    return encodings, names


def detect_face(frame, threshold=0.5):
    """
    Detect exactly one face and match against database.
    Returns name or None.
    """
    if frame is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")

    if len(locations) != 1:
        return None

    encs = face_recognition.face_encodings(rgb, locations)
    if not encs:
        return None

    db_enc, db_names = load_database()
    if not db_enc:
        return None

    distances = face_recognition.face_distance(db_enc, encs[0])
    idx = np.argmin(distances)

    if distances[idx] < threshold:
        return db_names[idx]

    return None


def register_face(frame, name):
    """
    Register a new face (exactly one face expected).
    """
    if frame is None:
        return False

    name = sanitize_name(name)
    if not name:
        return False

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")

    if len(locations) != 1:
        print("❌ Registration failed: need exactly one face")
        return False

    encs = face_recognition.face_encodings(rgb, locations)
    if not encs:
        return False

    person_dir = os.path.join(DB_PATH, name)
    os.makedirs(person_dir, exist_ok=True)

    file_path = os.path.join(person_dir, f"{len(os.listdir(person_dir)) + 1}.npy")
    np.save(file_path, encs[0])

    # Invalidate cache
    DB_CACHE["loaded"] = False

    print(f"✅ Registered {name}")
    return True