import cv2
import numpy as np
import os


def video_to_frames(video_path, max_frames=None):
    """读取视频并转换为RGB帧列表"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads in BGR, convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    return frames


def save_frames_to_video(frames, output_path, fps=30):
    """将RGB帧列表保存为视频文件"""
    if not frames:
        print("No frames to save.")
        return

    h, w, _ = frames[0].shape
    # OpenCV expects BGR
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()
    print(f"[Utils] Video saved to {output_path}")
