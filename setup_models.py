"""
SafeGear AI - Model Setup Script
Downloads required YOLO models automatically.

Run this script before first use:
    python setup_models.py
"""

import os
from pathlib import Path
from ultralytics import YOLO
import sys

def setup_models_directory():
    """Create models directory if it doesn't exist."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"✅ Models directory ready: {models_dir.absolute()}")
    return models_dir

def download_model(model_name: str, models_dir: Path):
    """Download YOLO model if not already present."""
    model_file = models_dir / f"{model_name}.pt"
    
    if model_file.exists():
        print(f"✅ Model already exists: {model_name}.pt")
        return str(model_file)
    
    print(f"⬇️ Downloading {model_name}...")
    try:
        # Download using ultralytics
        model = YOLO(model_name)
        
        # Move to models directory if downloaded elsewhere
        default_path = Path(f"{model_name}.pt")
        if default_path.exists() and not model_file.exists():
            default_path.rename(model_file)
        
        print(f"✅ Successfully downloaded: {model_name}.pt")
        return str(model_file)
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {str(e)}")
        return None

def main():
    """Main setup function."""
    print("=" * 60)
    print("🦺 SafeGear AI - Model Setup")
    print("=" * 60)
    print()
    
    # Setup directory
    models_dir = setup_models_directory()
    
    # Models to download
    models = [
        "yolov8n.pt",   # YOLOv8 Nano - Fastest, recommended for CPU
        "yolov8s.pt",   # YOLOv8 Small - Better accuracy
        "yolo11n.pt",   # YOLOv11 Nano - Latest version
    ]
    
    print(f"📦 Downloading {len(models)} models...")
    print()
    
    downloaded = []
    failed = []
    
    for model_name in models:
        result = download_model(model_name.replace('.pt', ''), models_dir)
        if result:
            downloaded.append(model_name)
        else:
            failed.append(model_name)
        print()
    
    # Summary
    print("=" * 60)
    print("📊 Setup Summary")
    print("=" * 60)
    print(f"✅ Downloaded: {len(downloaded)}/{len(models)}")
    print(f"❌ Failed: {len(failed)}/{len(models)}")
    
    if downloaded:
        print("\n📁 Downloaded models:")
        for model in downloaded:
            print(f"   • {model}")
    
    if failed:
        print("\n⚠️ Failed downloads:")
        for model in failed:
            print(f"   • {model}")
        print("\nYou can retry by running this script again.")
    
    print("\n" + "=" * 60)
    print("🚀 Ready to run: streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
