import numpy as np
from pathlib import Path
import os
import cv2
from skimage.feature import local_binary_pattern

def preprocess_image(image_input, size=(128, 128), to_gray=True):
    # Nếu là đường dẫn, đọc ảnh
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Không đọc được ảnh từ đường dẫn: {image_input}")
    # Nếu là ảnh ndarray, dùng trực tiếp
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise ValueError("Tham số truyền vào phải là đường dẫn hoặc ảnh numpy array")

    # Resize
    img = cv2.resize(img, size)

    # Convert to grayscale nếu cần
    if to_gray:
        if len(img.shape) == 2:
            return img
        elif len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Ảnh có số kênh không hợp lệ")
    return img

def extract_lbp_features(image_path, size=(128, 128), block_size=(16, 16), P=8, R=1, method='uniform'):
    gray = preprocess_image(image_path, size)

    lbp = local_binary_pattern(gray, P, R, method)

    h, w = size
    bh, bw = block_size
    blocks_y = h // bh
    blocks_x = w // bw

    n_bins=59

    features = []
    for i in range(blocks_y):
        for j in range(blocks_x):
            block = lbp[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            hist, _ = np.histogram(block.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype("float32")
            hist /= (hist.sum() + 1e-6)  # chuẩn hóa
            features.extend(hist)

    return np.array(features)


def load_lbp_features_from_folder(folder_path, size=(128, 128),
                                  block_size=(16, 16), P=8, R=1, method='uniform', max_images=None):
    X_list = []

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if max_images:
        files = files[:max_images]

    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            lbp_vector = extract_lbp_features(file_path)
            X_list.append(lbp_vector)
        except Exception as e:
            print(f"Lỗi xử lý ảnh {file_path}: {e}")

    X = np.array(X_list, dtype=np.float32)
    return X
