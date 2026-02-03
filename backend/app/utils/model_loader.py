"""Model loading utilities"""


class ModelLoader:
    """Load and cache models"""

    def __init__(self):
        self.models = {}

    def load(self, model_path: str, framework: str = "pytorch"):
        """Load model from path"""
        if model_path in self.models:
            return self.models[model_path]

        # Model loading handled in routes/inference.py
        self.models[model_path] = None
        return None


model_loader = ModelLoader()
