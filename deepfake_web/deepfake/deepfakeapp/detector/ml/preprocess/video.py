import cv2
from pathlib import Path

def save_frames(video_path, out_dir, prefix="frame"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (n_frames / fps) if fps > 0 else 0
    step_ms = 1000.0 if duration >= 10 else 500.0

    saved, ts = [], 0.0
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts)
        ok, frame = cap.read()
        if not ok:
            break
        p = out_dir / f"{prefix}_{len(saved):05d}.jpg"
        if cv2.imwrite(str(p), frame):
            saved.append(p)
        ts += step_ms

    cap.release()
    return saved
