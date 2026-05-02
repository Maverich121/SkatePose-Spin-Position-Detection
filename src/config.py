import os

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "video2.mp4")
PREDICTION_DATA_DIR = os.path.join(SCRIPT_DIR, "prediction_data")
FRAMES_DIR = os.path.join(PREDICTION_DATA_DIR, "prediction_frames")
POSE_DIR = os.path.join(PREDICTION_DATA_DIR, "prediction_frames_pose")
TRACKS_DIR = os.path.join(PREDICTION_DATA_DIR, "prediction_landmark_tracks")
