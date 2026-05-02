import cv2
import os
import sys

from . import config


def split_video():
    print("=" * 60)
    print("STEP 1: Splitting video into frames")
    print("=" * 60)
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: could not open video '{config.VIDEO_PATH}'")
        sys.exit(1)

    os.makedirs(config.FRAMES_DIR, exist_ok=True)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(config.FRAMES_DIR, f"frame_{count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()
    print(f"Extracted {count} frames to '{config.FRAMES_DIR}/'")
    return count
