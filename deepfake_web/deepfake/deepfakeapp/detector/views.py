from django.shortcuts import render
from .utils.file import create_tmp_dir, save_uploaded_file, media_url_from_path, clear_tmp_root
from .ml.preprocess.video import save_frames
from .ml.preprocess.image import crop_faces_in_folder
from .ml.preprocess.features import load_lbp_features_from_folder
from .ml.predict import predict_video_from_matrix
from django.contrib import messages
from pathlib import Path
from django.views.decorators.cache import never_cache
import os, shutil
from django.conf import settings


def home(request):
    context = {
        "real_url": "https://res.cloudinary.com/dnoubiojc/video/upload/v1755933960/real_inqlsu.mp4",
        "fake_url": "https://res.cloudinary.com/dnoubiojc/video/upload/v1755933960/fake_yxvjap.mp4",
    }
    return render(request, "home.html", context)

def contact(request):
    return render(request, "contact.html")

def detect(request):
    ctx = {}

    if request.method == "GET":
        clear_tmp_root()
        return render(request, "detect.html")

    if request.method == "POST" and request.FILES.getlist("videos"):
        files = request.FILES.getlist("videos")
        if len(files) > 10:
            ctx["error"] = "Bạn chỉ có thể upload tối đa 10 video!"
            return render(request, "detect.html", ctx)

        results = []
        clear_tmp_root()

        export_root = Path(settings.MEDIA_ROOT) / "export"
        shutil.rmtree(export_root, ignore_errors=True)
        export_root.mkdir(parents=True, exist_ok=True)

        counters = {"Real": 0, "Deepfake": 0}

        for f in files:
            # 1. Lưu video tạm
            tmp_dir = create_tmp_dir()
            saved_video_path = save_uploaded_file(f, tmp_dir)
            file_url = media_url_from_path(saved_video_path)

            # 2. Tách frame -> frames/
            frames_dir = tmp_dir / "frames"
            frame_paths = save_frames(video_path=saved_video_path, out_dir=frames_dir)

            # 3. Cắt mặt -> cropped/
            cropped_dir = tmp_dir / "cropped"
            cropped_paths = crop_faces_in_folder(src_dir=frames_dir, dst_dir=cropped_dir)

            # 4. LBP feature
            res = load_lbp_features_from_folder(cropped_dir)
            X = res

            # 5. Predict
            result, vote_conf = predict_video_from_matrix(X)

            results.append({
                "file_url": file_url,
                "result": result,
                "prob": vote_conf
            })

            # ====== Export ra thư mục theo nhãn ======
            counters[result] += 1
            label_dir = export_root / result
            label_dir.mkdir(parents=True, exist_ok=True)

            ext = Path(saved_video_path).suffix
            new_name = f"{result}_{counters[result]:02d}{ext}"
            shutil.copy(saved_video_path, label_dir / new_name)

        # Nén thư mục export thành .zip
        zip_path = shutil.make_archive(str(export_root), "zip", export_root)
        zip_path = Path(zip_path)
        rel_path = zip_path.relative_to(settings.MEDIA_ROOT)

        ctx["results"] = results
        ctx["zip_url"] = f"{settings.MEDIA_URL}{rel_path.as_posix()}"
        return render(request, "detect.html", ctx)

    return render(request, "detect.html", ctx)



def about(request):
    return render(request, "about.html")
