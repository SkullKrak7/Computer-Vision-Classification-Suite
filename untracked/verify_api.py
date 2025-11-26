#!/usr/bin/env python3
"""Verify all models are accessible via API"""

import json
from pathlib import Path

def verify_models():
    """Verify all model files and metadata exist"""
    models_dir = Path("models")
    
    models = {
        "KNN": {
            "model": "knn_model.pkl",
            "metadata": "knn_metadata.json"
        },
        "SVM": {
            "model": "svm_model.pkl",
            "metadata": "svm_metadata.json"
        },
        "PyTorch CNN": {
            "model": "pytorch_cnn_tuned.pth",
            "metadata": "pytorch_cnn_tuned_metadata.json"
        },
        "TensorFlow MobileNetV2": {
            "model": "tensorflow_mobilenet_tuned.keras",
            "metadata": "tensorflow_mobilenet_tuned_metadata.json"
        }
    }
    
    print("=" * 70)
    print("MODEL VERIFICATION REPORT")
    print("=" * 70)
    
    all_good = True
    results = []
    
    for name, files in models.items():
        model_path = models_dir / files["model"]
        metadata_path = models_dir / files["metadata"]
        
        model_exists = model_path.exists()
        metadata_exists = metadata_path.exists()
        
        status = "✅" if (model_exists and metadata_exists) else "❌"
        
        result = {
            "name": name,
            "status": status,
            "model_exists": model_exists,
            "metadata_exists": metadata_exists
        }
        
        if metadata_exists:
            with open(metadata_path) as f:
                data = json.load(f)
                result["accuracy"] = f"{data['accuracy']*100:.2f}%"
                result["f1_score"] = f"{data['f1_score']*100:.2f}%"
        
        results.append(result)
        
        if not (model_exists and metadata_exists):
            all_good = False
    
    # Print results
    for r in results:
        print(f"\n{r['status']} {r['name']}")
        print(f"   Model file: {'✓' if r['model_exists'] else '✗'}")
        print(f"   Metadata:   {'✓' if r['metadata_exists'] else '✗'}")
        if 'accuracy' in r:
            print(f"   Accuracy:   {r['accuracy']}")
            print(f"   F1 Score:   {r['f1_score']}")
    
    print("\n" + "=" * 70)
    if all_good:
        print("✅ ALL MODELS VERIFIED - READY FOR DEPLOYMENT")
    else:
        print("❌ SOME MODELS MISSING - CHECK ABOVE")
    print("=" * 70)
    
    return all_good

if __name__ == "__main__":
    verify_models()
