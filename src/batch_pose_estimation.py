import cv2
import mediapipe as mp
import os
import glob

from utils.pose_estimation import download_model, draw_landmarks, MODEL_PATH

from . import config

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def batch_pose_estimation():
    print("\n" + "=" * 60)
    print("STEP 2: Running pose estimation on each frame")
    print("=" * 60)
    download_model()
    os.makedirs(config.POSE_DIR, exist_ok=True)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_paths = sorted(glob.glob(os.path.join(config.FRAMES_DIR, "*.jpg")))
    total = len(frame_paths)

    with PoseLandmarker.create_from_options(options) as landmarker:
        for i, frame_path in enumerate(frame_paths):
            filename = os.path.basename(frame_path)
            image = cv2.imread(frame_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = landmarker.detect(mp_image)

            annotated = image.copy()
            if result.pose_landmarks:
                draw_landmarks(annotated, result.pose_landmarks[0])

            cv2.imwrite(os.path.join(config.POSE_DIR, filename), annotated)

            if (i + 1) % 20 == 0 or (i + 1) == total:
                print(f"[{i+1}/{total}] Done")

    print(f"Annotated frames saved to '{config.POSE_DIR}/'")