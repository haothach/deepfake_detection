# detector/ml/predict.py
from pathlib import Path
import numpy as np
import joblib
from django.conf import settings

MODEL = None
CLASSES = None

def load_model():
    global MODEL, CLASSES
    if MODEL is None:
        model_path = Path(settings.BASE_DIR) / "detector/ml/models/best_pipe.pkl"
        MODEL = joblib.load(model_path)
        CLASSES = getattr(MODEL, "classes_", None)
    return MODEL

def map_label(y):
    # Chuẩn hoá nhãn về "Real"/"Deepfake"
    if isinstance(y, str):
        return y
    if y in (0, 1):
        return "Deepfake" if y == 1 else "Real"
    return str(y)

def predict_video_from_matrix(X: np.ndarray):
    if X is None or X.size == 0:
        return "No face/feature", 0.0, {}

    model = load_model()

    # Dự đoán tất cả khung hình 1 lượt
    try:
        y_pred = model.predict(X)  # (N,)
        labels = [map_label(y) for y in y_pred]
    except Exception as e:
        raise RuntimeError(f"Predict failed: {e}")

    if len(labels) == 0:
        return "No prediction", 0.0, {}

    # Đa số phiếu
    uniq, counts = np.unique(labels, return_counts=True)
    winner = uniq[counts.argmax()]
    conf = counts.max() / counts.sum() * 100.0

    return winner, round(conf, 2)
