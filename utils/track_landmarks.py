import cv2
import mediapipe as mp
import os
import glob
import json
import csv
import colorsys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.pose_estimation import (
    download_model,
    MODEL_PATH,
    LANDMARK_NAMES,
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(SCRIPT_DIR, "frames")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "landmark_tracks")


def _generate_colors_rgb(n):
    colors = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 0.85)
        colors.append((r, g, b))
    return colors


def extract_all_landmarks(frame_paths):
    """Run pose estimation on every frame and return per-landmark tracks.

    Returns:
        tracks: dict  {landmark_index: {"x": [...], "y": [...], "frame": [...]}}
        img_w, img_h: image dimensions (pixels)
    """
    download_model()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    num_landmarks = len(LANDMARK_NAMES)
    tracks = {i: {"x": [], "y": [], "frame": []} for i in range(num_landmarks)}
    img_w, img_h = None, None
    total = len(frame_paths)

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

    return tracks, img_w, img_h


def save_csv(tracks, output_path):
    """Save all landmark tracks to a single CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["landmark", "frame", "x", "y"])
        for idx in sorted(tracks.keys()):
            name = LANDMARK_NAMES[idx]
            for frame, x, y in zip(tracks[idx]["frame"], tracks[idx]["x"], tracks[idx]["y"]):
                writer.writerow([name, frame, f"{x:.2f}", f"{y:.2f}"])
    print(f"CSV saved to {output_path}")


def save_json(tracks, output_path):
    """Save all landmark tracks to a JSON file."""
    data = {}
    for idx in sorted(tracks.keys()):
        name = LANDMARK_NAMES[idx]
        data[name] = {
            "frames": tracks[idx]["frame"],
            "x": [round(v, 2) for v in tracks[idx]["x"]],
            "y": [round(v, 2) for v in tracks[idx]["y"]],
        }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON saved to {output_path}")


def plot_x_over_time(tracks, output_path, colors):
    """Plot x-coordinate of each landmark over frames."""
    fig, ax = plt.subplots(figsize=(14, 7))
    for idx in sorted(tracks.keys()):
        if not tracks[idx]["frame"]:
            continue
        ax.plot(tracks[idx]["frame"], tracks[idx]["x"],
                color=colors[idx], label=LANDMARK_NAMES[idx], linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("X (px)")
    ax.set_title("Landmark X position over time")
    ax.legend(fontsize=5, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"X-plot saved to {output_path}")


def plot_y_over_time(tracks, output_path, colors):
    """Plot y-coordinate of each landmark over frames."""
    fig, ax = plt.subplots(figsize=(14, 7))
    for idx in sorted(tracks.keys()):
        if not tracks[idx]["frame"]:
            continue
        ax.plot(tracks[idx]["frame"], tracks[idx]["y"],
                color=colors[idx], label=LANDMARK_NAMES[idx], linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Y (px)")
    ax.set_title("Landmark Y position over time")
    ax.legend(fontsize=5, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Y-plot saved to {output_path}")


def _filter_range(track, start, end):
    """Return lists filtered to frames in [start, end]."""
    xs, ys, fs = [], [], []
    for f, x, y in zip(track["frame"], track["x"], track["y"]):
        if start <= f <= end:
            xs.append(x)
            ys.append(y)
            fs.append(f)
    return xs, ys, fs


def plot_per_landmark(tracks, output_dir, colors, img_w, img_h,
                      start_frame=None, end_frame=None):
    """One plot per landmark showing its 2D trajectory on the image plane.
    If start_frame/end_frame are given, full trajectory is drawn faded and
    the selected range is highlighted."""
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    highlight = start_frame is not None and end_frame is not None

    for idx in sorted(tracks.keys()):
        if not tracks[idx]["frame"]:
            continue
        name = LANDMARK_NAMES[idx]
        fig, ax = plt.subplots(figsize=(7, 7))
        xs = tracks[idx]["x"]
        ys = tracks[idx]["y"]

        # Full trajectory (faded when highlighting a sub-range)
        alpha_full = 0.15 if highlight else 0.5
        ax.plot(xs, ys, color=colors[idx], linewidth=0.5, alpha=alpha_full, zorder=1)
        ax.scatter(xs, ys, c="lightgrey" if highlight else range(len(xs)),
                   cmap=None if highlight else "viridis", s=4, zorder=2, alpha=0.3 if highlight else 1.0)

        if highlight:
            hx, hy, hf = _filter_range(tracks[idx], start_frame, end_frame)
            if hx:
                ax.plot(hx, hy, color=colors[idx], linewidth=1.5, alpha=0.9, zorder=3)
                sc = ax.scatter(hx, hy, c=range(len(hx)), cmap="plasma", s=14, zorder=4,
                                edgecolors="black", linewidths=0.3)
                # Mark start and end
                ax.scatter([hx[0]], [hy[0]], marker="o", s=80, color="lime",
                           edgecolors="black", linewidths=1.2, zorder=5, label="start")
                ax.scatter([hx[-1]], [hy[-1]], marker="X", s=80, color="red",
                           edgecolors="black", linewidths=1.2, zorder=5, label="end")
                ax.legend(fontsize=8)

        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)  # invert y to match image coords
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        suffix = f" (frames {start_frame}-{end_frame})" if highlight else ""
        ax.set_title(f"{name} trajectory{suffix}")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(os.path.join(traj_dir, f"{name}.png"), dpi=120)
        plt.close(fig)
    print(f"Per-landmark trajectory plots saved to {traj_dir}/")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    if not frame_paths:
        print("No frames found in", FRAMES_DIR)
        return

    print(f"Found {len(frame_paths)} frames. Extracting landmarks...")
    tracks, img_w, img_h = extract_all_landmarks(frame_paths)

    colors = _generate_colors_rgb(len(LANDMARK_NAMES))

    # Save raw data
    save_csv(tracks, os.path.join(OUTPUT_DIR, "landmark_tracks.csv"))
    save_json(tracks, os.path.join(OUTPUT_DIR, "landmark_tracks.json"))

    # Summary plots
    plot_x_over_time(tracks, os.path.join(OUTPUT_DIR, "x_over_time.png"), colors)
    plot_y_over_time(tracks, os.path.join(OUTPUT_DIR, "y_over_time.png"), colors)

    # Per-landmark 2D trajectory (highlight frames 60–145)
    plot_per_landmark(tracks, OUTPUT_DIR, colors, img_w, img_h,
                      start_frame=60, end_frame=145)

    print(f"\nAll outputs saved under '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
