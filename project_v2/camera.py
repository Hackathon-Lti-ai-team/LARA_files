# camera.py
# Camera utility: short-lived, no UI, Pi-safe

import cv2
import time

class Camera:
    def __init__(self, device="/dev/video0", width=1280, height=720, fps=30):
        self.device = device
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Cannot open camera {device}")

        # Force MJPEG for Pi performance
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"📷 Camera opened: {device} ({actual_w}x{actual_h} @ {actual_fps}fps)")

    def read(self):
        """Read single frame."""
        return self.cap.read()

    def capture_best_frame(self, duration_sec=5):
        """
        Capture frames for duration_sec and return the last valid frame.
        Used for face recognition.
        """
        best_frame = None
        start = time.time()

        while time.time() - start < duration_sec:
            ret, frame = self.cap.read()
            if ret:
                best_frame = frame

        return best_frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None