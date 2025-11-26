"""Inference API routes"""

import io
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from ..models import InferenceResponse

router = APIRouter()

# Load PyTorch model (best performing)
model = None
label_map = None


def load_model():
    global model, label_map
    if model is None:
        try:
            model_path = Path("models/pytorch_cnn_tuned.pth")
            if not model_path.exists():
                return None

            checkpoint = torch.load(model_path, map_location="cpu")
            
            # Handle checkpoint format - could be dict or OrderedDict
            if isinstance(checkpoint, dict) and any(k in checkpoint for k in ["model_state", "model_state_dict"]):
                state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
                label_map = checkpoint.get("label_map", {str(i): i for i in range(6)})
            else:
                # Direct state_dict
                state_dict = checkpoint
                label_map = {str(i): i for i in range(6)}

            # Model architecture with BatchNorm (matching saved weights)
            from torch import nn

            class SimpleCNN(nn.Module):
                def __init__(self, num_classes=6):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),           # 0
                        nn.BatchNorm2d(32),                        # 1
                        nn.ReLU(),                                 # 2
                        nn.MaxPool2d(2, 2),                        # 3
                        nn.Dropout2d(0.2),                         # 4
                        nn.Conv2d(32, 64, 3, padding=1),          # 5
                        nn.BatchNorm2d(64),                        # 6
                        nn.ReLU(),                                 # 7
                        nn.MaxPool2d(2, 2),                        # 8
                        nn.Dropout2d(0.3),                         # 9
                        nn.Conv2d(64, 128, 3, padding=1),         # 10
                        nn.BatchNorm2d(128),                       # 11
                        nn.ReLU(),                                 # 12
                        nn.MaxPool2d(2, 2)                         # 13
                    )
                    self.classifier = nn.Sequential(
                        nn.Flatten(),                              # 0
                        nn.Linear(128 * 8 * 8, 256),              # 1 - trained on 64x64 images
                        nn.BatchNorm1d(256),                       # 2
                        nn.ReLU(),                                 # 3
                        nn.Dropout(0.5),                           # 4
                        nn.Linear(256, num_classes)                # 5
                    )

                def forward(self, x):
                    return self.classifier(self.features(x))

            model = SimpleCNN(num_classes=len(label_map))
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            print(f"Model load error: {e}")
            import traceback
            traceback.print_exc()
            return None
    return model


@router.post("/predict", response_model=InferenceResponse)
async def predict(file: UploadFile = File(...)):
    """Run inference on uploaded image"""
    start = time.time()

    if load_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read and preprocess image (model trained on 64x64)
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((64, 64))
        img_array = np.array(image) / 255.0
        img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top3 = torch.topk(probs, min(3, len(probs)))

        # Map class indices to names
        class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
        predictions = [
            {
                "class_id": idx.item(),
                "class_name": class_names[idx.item()] if idx.item() < len(class_names) else f"class_{idx.item()}",
                "confidence": prob.item()
            }
            for prob, idx in zip(top3.values, top3.indices)
        ]

        return InferenceResponse(predictions=predictions, inference_time=time.time() - start)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
