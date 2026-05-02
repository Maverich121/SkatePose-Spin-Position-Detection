import cv2
import os
import sys


def split_video(video_path: str, output_dir: str = "frames") -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video '{video_path}'")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()
    print(f"Extracted {count} frames to '{output_dir}/'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_frames.py <video_path> [output_dir]")
        sys.exit(1)

    video = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "frames"
    split_video(video, out_dir)
