from pathlib import Path
from uuid import uuid4
from django.conf import settings
import shutil

def clear_tmp_root():
    root = Path(settings.DETECT_TMP_ROOT)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return
    for p in root.iterdir():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)

def create_tmp_dir():
    tmp_dir = Path(settings.DETECT_TMP_ROOT) / uuid4().hex
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir

def save_uploaded_file(uploaded_file, folder: Path):
    ext = Path(uploaded_file.name).suffix or ".mp4"
    out_path = folder / f"video{ext}"
    with open(out_path, "wb") as f:
        for x in uploaded_file.chunks():
            f.write(x)
    return out_path

def media_url_from_path(abs_path: Path):
    rel = abs_path.relative_to(Path(settings.MEDIA_ROOT)).as_posix()
    return f"{settings.MEDIA_URL}{rel}"
