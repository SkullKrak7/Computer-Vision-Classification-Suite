"""Image preprocessing utilities"""

import cv2
import numpy as np

def preprocess_image(image_bytes: bytes, size: tuple = (224, 224)) -> np.ndarray:
    """Preprocess image for inference"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return img
