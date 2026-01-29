# face_draw.py
# Fast face centering logic (30 FPS + flipped view)

import cv2
import face_recognition


# ---------------- CONFIG ----------------
CAMERA_INDEX = 0

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

TARGET_FPS = 30
FRAME_SKIP = 2            # run face detection every N frames

BOX_MARGIN = 60
CENTER_TOLERANCE = 80


def main():
    print("📸 Face centering started (press 'q' to quit)")

    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Camera configuration
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    frame_count = 0
    last_locations = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally (mirror correction)
        frame = cv2.flip(frame, 1)

        # Force consistent window size
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

        h, w, _ = frame.shape
        frame_center_x = w // 2  # imaginary center

        instruction = "NO FACE"

        # Run face detection only every N frames
        if frame_count % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_locations = face_recognition.face_locations(
                rgb,
                model="hog"
            )

        frame_count += 1

        if len(last_locations) == 1:
            top, right, bottom, left = last_locations[0]

            # Enlarge bounding box
            left = max(0, left - BOX_MARGIN)
            top = max(0, top - BOX_MARGIN)
            right = min(w, right + BOX_MARGIN)
            bottom = min(h, bottom + BOX_MARGIN)

            # Draw face bounding box
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            # Face center
            face_center_x = (left + right) // 2
            face_center_y = (top + bottom) // 2
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)

            dx = face_center_x - frame_center_x

            if abs(dx) <= CENTER_TOLERANCE:
                instruction = "CENTERED"
            elif dx < 0:
                instruction = "MOVE RIGHT"
            else:
                instruction = "MOVE LEFT"

        # Display instruction
        cv2.putText(
            frame,
            instruction,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3
        )

        print(instruction)

        cv2.imshow("Face Centering", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Face centering stopped")


if __name__ == "__main__":
    main()
