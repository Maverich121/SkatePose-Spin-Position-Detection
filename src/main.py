"""
Full pipeline for prediction_target.mp4:
  1. Split video into frames           -> prediction_frames/
  2. Run pose estimation on each frame  -> prediction_frames_pose/
  3. Extract landmark tracks (CSV/JSON) -> prediction_landmark_tracks/
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.split_video import split_video
from src.batch_pose_estimation import batch_pose_estimation
from src.extract_landmark_tracks import extract_landmark_tracks


def clear_output_folders():
    for folder in [config.FRAMES_DIR, config.POSE_DIR, config.TRACKS_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Cleared: {folder}/")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config.VIDEO_PATH = sys.argv[1]

    if not os.path.exists(config.VIDEO_PATH):
        print(f"Error: video not found at '{config.VIDEO_PATH}'")
        sys.exit(1)

    clear_output_folders()

    split_video()
    batch_pose_estimation()
    extract_landmark_tracks()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Frames:          {config.FRAMES_DIR}/")
    print(f"  Annotated poses: {config.POSE_DIR}/")
    print(f"  Landmark tracks: {config.TRACKS_DIR}/")
    print("=" * 60)
