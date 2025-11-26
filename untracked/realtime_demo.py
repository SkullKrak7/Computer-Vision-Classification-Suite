#!/usr/bin/env python3
"""Real-time classification demo"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.deep_learning import PyTorchCNNClassifier


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/pytorch/best_model.pth"
    model = PyTorchCNNClassifier.load(model_path)

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        img = cv2.resize(frame, (224, 224))
        img = img.astype(np.float32) / 255.0

        # Predict
        pred = model.predict(np.expand_dims(img, 0))[0]
        label = model.label_map[pred]

        # Display
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
