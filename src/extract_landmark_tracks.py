import cv2
import mediapipe as mp
import os
import glob
import json
import csv

from utils.pose_estimation import download_model, MODEL_PATH, LANDMARK_NAMES

from . import config

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def extract_landmark_tracks():
    print("\n" + "=" * 60)
    print("STEP 3: Extracting landmark tracks")
    print("=" * 60)
    download_model()
    os.makedirs(config.TRACKS_DIR, exist_ok=True)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_paths = sorted(glob.glob(os.path.join(config.FRAMES_DIR, "*.jpg")))
    total = len(frame_paths)
    num_landmarks = len(LANDMARK_NAMES)
    tracks = {i: {"x": [], "y": [], "frame": []} for i in range(num_landmarks)}
    img_w, img_h = None, None

    with PoseLandmarker.create_from_options(options) as landmarker:
        for fi, frame_path in enumerate(frame_paths):
            image = cv2.imread(frame_path)
            if image is None:
                continue
            if img_w is None:
                img_h, img_w = image.shape[:2]

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = landmarker.detect(mp_image)

            if not result.pose_landmarks:
                continue

            landmarks = result.pose_landmarks[0]
            for idx, lm in enumerate(landmarks):
                if idx >= num_landmarks:
                    break
                tracks[idx]["x"].append(lm.x * img_w)
                tracks[idx]["y"].append(lm.y * img_h)
                tracks[idx]["frame"].append(fi)

            if (fi + 1) % 20 == 0 or (fi + 1) == total:
                print(f"[{fi+1}/{total}] Extracted")

    # Save CSV
    csv_path = os.path.join(config.TRACKS_DIR, "landmark_tracks.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["landmark", "frame", "x", "y"])
        for idx in sorted(tracks.keys()):
            name = LANDMARK_NAMES[idx]
            for frame, x, y in zip(tracks[idx]["frame"], tracks[idx]["x"], tracks[idx]["y"]):
                writer.writerow([name, frame, f"{x:.2f}", f"{y:.2f}"])
    print(f"CSV saved to {csv_path}")

    # Save JSON
    json_path = os.path.join(config.TRACKS_DIR, "landmark_tracks.json")
    data = {}
    for idx in sorted(tracks.keys()):
        name = LANDMARK_NAMES[idx]
        data[name] = {
            "frames": tracks[idx]["frame"],
            "x": [round(v, 2) for v in tracks[idx]["x"]],
            "y": [round(v, 2) for v in tracks[idx]["y"]],
        }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON saved to {json_path}")
