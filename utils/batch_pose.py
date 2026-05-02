import cv2
import mediapipe as mp
import os
import glob

from utils.pose_estimation import (
    download_model,
    draw_landmarks,
    MODEL_PATH,
    LANDMARK_NAMES,
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames_pose")


def batch_pose_estimation():
    download_model()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    total = len(frame_paths)
    print(f"Processing {total} frames...")

    with PoseLandmarker.create_from_options(options) as landmarker:
        for i, frame_path in enumerate(frame_paths):
            filename = os.path.basename(frame_path)
            image = cv2.imread(frame_path)
            if image is None:
                print(f"[{i+1}/{total}] Skipping {filename} (unreadable)")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = landmarker.detect(mp_image)

            annotated = image.copy()
            if result.pose_landmarks:
                draw_landmarks(annotated, result.pose_landmarks[0])

            output_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(output_path, annotated)

            if (i + 1) % 20 == 0 or (i + 1) == total:
                print(f"[{i+1}/{total}] Done")

    print(f"Annotated frames saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    batch_pose_estimation()
