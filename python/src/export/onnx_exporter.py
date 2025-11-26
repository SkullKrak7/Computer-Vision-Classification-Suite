"""
ONNX model export utilities
Converts trained models to ONNX format for C++ inference
"""

import json
import logging
from pathlib import Path

import tensorflow as tf
import torch

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export trained models to ONNX format"""

    @staticmethod
    def export_pytorch(model_path: str, output_path: str, input_shape: tuple = (1, 3, 224, 224)):
        """
        Export PyTorch model to ONNX

        Args:
            model_path: Path to .pth file
            output_path: Path for .onnx output
            input_shape: Input tensor shape (batch, channels, height, width)
        """
        from src.models.deep_learning.pytorch_cnn import PyTorchCNNClassifier

        logger.info(f"Loading PyTorch model from {model_path}")
        classifier = PyTorchCNNClassifier.load(model_path)

        classifier.model.eval()

        dummy_input = torch.randn(input_shape)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            classifier.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        ONNXExporter._save_metadata(output_path, classifier.label_map, input_shape)

        logger.info(f"PyTorch model exported to {output_path}")

    @staticmethod
    def export_tensorflow(model_path: str, output_path: str):
        """
        Export TensorFlow model to ONNX

        Args:
            model_path: Path to .h5 file
            output_path: Path for .onnx output
        """
        try:
            import tf2onnx
        except ImportError:
            raise ImportError("tf2onnx is required. Install with: pip install tf2onnx")

        from src.models.deep_learning.tf_mobilenet import TFMobileNetClassifier

        logger.info(f"Loading TensorFlow model from {model_path}")
        classifier = TFMobileNetClassifier.load(model_path)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        spec = (tf.TensorSpec(classifier.model.input_shape, tf.float32, name="input"),)

        model_proto, _ = tf2onnx.convert.from_keras(
            classifier.model, input_signature=spec, opset=11
        )

        with open(output_path, "wb") as f:
            f.write(model_proto.SerializeToString())

        ONNXExporter._save_metadata(output_path, classifier.label_map, classifier.input_shape)

        logger.info(f"TensorFlow model exported to {output_path}")

    @staticmethod
    def _save_metadata(onnx_path: Path, label_map: dict, input_shape: tuple):
        """Save label map and metadata alongside ONNX model"""
        metadata_path = onnx_path.parent / f"{onnx_path.stem}_metadata.json"

        with open(metadata_path, "w") as f:
            json.dump({"label_map": label_map, "input_shape": input_shape}, f, indent=2)

        labels_path = onnx_path.parent / "labels.txt"
        with open(labels_path, "w") as f:
            for idx in sorted(label_map.keys()):
                f.write(f"{label_map[idx]}\n")

        logger.info(f"Metadata saved to {metadata_path}")


def export_model(model_path: str, output_dir: str = "models/onnx", framework: str = "auto"):
    """
    Convenience function to export any model to ONNX

    Args:
        model_path: Path to trained model
        output_dir: Output directory for ONNX file
        framework: 'pytorch', 'tensorflow', or 'auto' to detect
    """
    model_path = Path(model_path)

    if framework == "auto":
        if model_path.suffix == ".pth":
            framework = "pytorch"
        elif model_path.suffix == ".h5":
            framework = "tensorflow"
        else:
            raise ValueError(f"Cannot auto-detect framework from {model_path.suffix}")

    output_path = Path(output_dir) / f"{model_path.stem}.onnx"

    if framework == "pytorch":
        ONNXExporter.export_pytorch(str(model_path), str(output_path))
    elif framework == "tensorflow":
        ONNXExporter.export_tensorflow(str(model_path), str(output_path))
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export trained models to ONNX")
    parser.add_argument("model_path", type=str, help="Path to trained model file")
    parser.add_argument(
        "--output-dir", type=str, default="models/onnx", help="Output directory for ONNX file"
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=["auto", "pytorch", "tensorflow"],
        default="auto",
        help="Framework of the model",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_path = export_model(args.model_path, args.output_dir, args.framework)
    print(f"\nModel exported to: {output_path}")
