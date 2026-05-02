import cv2
import mediapipe as mp
import sys
import os
import urllib.request
import colorsys

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_heavy.task")

# 33 pose landmark names
LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

# Pose skeleton connections (pairs of landmark indices)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),   # left eye
    (0, 4), (4, 5), (5, 6), (6, 8),   # right eye
    (9, 10),                            # mouth
    (11, 12),                           # shoulders
    (11, 13), (13, 15),                 # left arm
    (12, 14), (14, 16),                 # right arm
    (15, 17), (15, 19), (15, 21),       # left hand
    (16, 18), (16, 20), (16, 22),       # right hand
    (11, 23), (12, 24),                 # torso
    (23, 24),                           # hips
    (23, 25), (25, 27),                 # left leg
    (24, 26), (26, 28),                 # right leg
    (27, 29), (29, 31), (27, 31),       # left foot
    (28, 30), (30, 32), (28, 32),       # right foot
]


def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


def _generate_landmark_colors(n):
    """Generate n visually distinct colors (BGR) using evenly spaced hues."""
    colors = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colors


LANDMARK_COLORS = _generate_landmark_colors(len(LANDMARK_NAMES))


def draw_landmarks(image, landmarks):
    h, w, _ = image.shape
    points = []
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))

    # Draw connections (use the color of the starting landmark)
    for start, end in POSE_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(image, points[start], points[end], LANDMARK_COLORS[start], 2)

    # Draw landmark points, each with its own unique color
    for idx, (px, py) in enumerate(points):
        color = LANDMARK_COLORS[idx] if idx < len(LANDMARK_COLORS) else (0, 0, 255)
        cv2.circle(image, (px, py), 5, color, -1)


def estimate_pose(image_path: str, output_path: str = "output.jpg") -> None:
    download_model()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not read image '{image_path}'")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    with PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

        if not result.pose_landmarks:
            print("No pose detected in the image.")
            sys.exit(0)

        landmarks = result.pose_landmarks[0]

        # Print landmark coordinates
        for idx, lm in enumerate(landmarks):
            name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"LANDMARK_{idx}"
            print(f"{name:30s}  x={lm.x:.4f}  y={lm.y:.4f}  z={lm.z:.4f}  vis={lm.visibility:.2f}")

        # Draw on image and save
        annotated = image.copy()
        draw_landmarks(annotated, landmarks)
        cv2.imwrite(output_path, annotated)
        print(f"\nAnnotated image saved to '{output_path}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pose_estimation.py <image_path> [output_path]")
        sys.exit(1)

    img = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "output.jpg"
    estimate_pose(img, out)
