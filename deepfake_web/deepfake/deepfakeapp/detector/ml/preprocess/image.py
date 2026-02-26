import cv2
from pathlib import Path

HAAR_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def crop_and_resize_face(image_bgr, size=(128, 128)):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = HAAR_FACE.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])  # mặt lớn nhất
    face = image_bgr[y:y+h, x:x+w]
    return cv2.resize(face, size)

def crop_faces_in_folder(src_dir, dst_dir, size=(128, 128)):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [p for p in src_dir.iterdir() if p.suffix.lower() in exts]

    saved = []
    for idx, p in enumerate(sorted(files)):
        img = cv2.imread(str(p))
        if img is None:
            continue
        face = crop_and_resize_face(img, size=size)
        if face is None:
            continue
        out_path = dst_dir / f"face_{idx:05d}.jpg"
        if cv2.imwrite(str(out_path), face):
            saved.append(out_path)
    return saved
