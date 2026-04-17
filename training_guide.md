# SafeGear AI - Custom Model Training Guide

This guide explains how to train custom YOLO models for specific safety gear detection scenarios.

---

## 📋 Table of Contents

1. [Dataset Preparation](#dataset-preparation)
2. [Labeling Guidelines](#labeling-guidelines)
3. [Training Configuration](#training-configuration)
4. [Training Execution](#training-execution)
5. [Model Evaluation](#model-evaluation)
6. [Deployment](#deployment)

---

## Dataset Preparation

### Recommended Dataset Structure

```
dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       ├── img002.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Class Definitions

Create `dataset.yaml`:

```yaml
path: ./dataset
train: train/images
val: valid/images
test: test/images

nc: 6  # number of classes
names:
  0: Person
  1: Helmet
  2: No-Helmet
  3: Safety-Vest
  4: No-Vest
  5: Hard-Hat
```

### Data Collection Tips

| Environment | Min Images | Augmentation |
|-------------|------------|--------------|
| Construction sites | 500+ | Lighting, rotation |
| Road traffic | 1000+ | Weather conditions |
| Industrial | 300+ | Occlusion, distance |
| Healthcare | 400+ | Mask types, angles |

---

## Labeling Guidelines

### Bounding Box Standards

```
YOLO Format: <class_id> <x_center> <y_center> <width> <height>
```

**Best Practices:**
1. **Tight bounding boxes** - Include only the object, minimal background
2. **Consistent labeling** - Same class should have similar box sizes
3. **Occlusion handling** - Label partially visible objects if >20% visible
4. **Class balance** - Ensure similar counts for each class

### Example Label Files

**Image: worker.jpg**
```
0 0.500000 0.600000 0.300000 0.500000    # Person
1 0.480000 0.350000 0.150000 0.120000    # Helmet
3 0.500000 0.550000 0.200000 0.250000    # Safety-Vest
```

### Labeling Tools

| Tool | Platform | Best For |
|------|----------|----------|
| [Roboflow](https://roboflow.com) | Web | Team collaboration |
| [LabelImg](https://github.com/tzutalin/labelImg) | Desktop | Quick labeling |
| [CVAT](https://cvat.org) | Web | Complex projects |
| [MakeSense](https://www.makesense.ai) | Web | Beginners |

---

## Training Configuration

### Hardware Requirements

| Setup | GPU | RAM | Training Time (100 epochs) |
|-------|-----|-----|---------------------------|
| Minimum | None (CPU) | 8GB | 8-12 hours |
| Recommended | GTX 1660 Ti | 16GB | 2-3 hours |
| Optimal | RTX 3080 | 32GB | 30-45 minutes |

### Training Script

Create `train.py`:

```python
from ultralytics import YOLO

def train_safety_model():
    # Load pre-trained model
    model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt
    
    # Training configuration
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        workers=8,
        device=0,  # GPU device, use 'cpu' for CPU training
        patience=20,  # Early stopping patience
        save=True,
        project='safear_models',
        name='safety_gear_v1',
        
        # Augmentation
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation
        hsv_v=0.4,    # HSV-Value
        degrees=5,    # Rotation
        translate=0.1,  # Translation
        scale=0.5,    # Scale
        shear=2,      # Shear
        flipud=0.0,   # Flip up-down
        fliplr=0.5,   # Flip left-right
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.0,    # Mixup augmentation
        
        # Optimization
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )
    
    print(f"Training complete! Best model: {results.best}")
    return results

if __name__ == "__main__":
    train_safety_model()
```

### Hyperparameter Tuning

| Parameter | Default | Effect | Recommended Range |
|-----------|---------|--------|-------------------|
| `epochs` | 100 | Training iterations | 50-200 |
| `batch` | 16 | Images per batch | 8-32 (based on GPU) |
| `imgsz` | 640 | Input resolution | 416-1280 |
| `lr0` | 0.01 | Initial learning rate | 0.001-0.01 |
| `patience` | 50 | Early stopping | 10-50 |

---

## Training Execution

### Single GPU Training

```bash
python train.py
```

### Multi-GPU Training

```bash
python -m torch.distributed.run --nproc_per_node 2 train.py
```

### Resume Training

```python
model = YOLO('safear_models/safety_gear_v1/weights/last.pt')
model.train(resume=True)
```

---

## Model Evaluation

### Validation

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('safear_models/safety_gear_v1/weights/best.pt')

# Validate
metrics = model.val(data='dataset.yaml')

print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")
```

### Class-wise Performance

```python
# Per-class metrics
for i, name in enumerate(metrics.names):
    print(f"{name}:")
    print(f"  Precision: {metrics.box.precision[i]:.3f}")
    print(f"  Recall: {metrics.box.recall[i]:.3f}")
    print(f"  mAP50: {metrics.box.ap50[i]:.3f}")
```

### Export to Production Formats

```python
# ONNX format
model.export(format='onnx', imgsz=640)

# TensorRT (for NVIDIA GPUs)
model.export(format='engine', imgsz=640, half=True)

# OpenVINO (for Intel CPUs)
model.export(format='openvino', imgsz=640)

# CoreML (for Apple devices)
model.export(format='coreml', imgsz=640)
```

---

## Deployment

### Integration with SafeGear AI

1. **Copy model to models directory:**
   ```bash
   cp safear_models/safety_gear_v1/weights/best.pt models/custom_safety.pt
   ```

2. **Update config.py:**
   ```python
   MODEL_CONFIG['custom'] = {
       'name': 'Custom Safety Model',
       'speed': 'Medium',
       'accuracy': 'Custom',
       'model_path': 'models/custom_safety.pt',
       'recommended': False
   }
   ```

3. **Update app.py model selection:**
   The custom model will appear in the Streamlit sidebar automatically.

### Performance Optimization

| Technique | Speed Gain | Quality Impact |
|-----------|------------|----------------|
| INT8 Quantization | 2-3x faster | Minimal loss |
| Half Precision (FP16) | 1.5x faster | No loss on GPUs |
| Batch Inference | Variable | None |
| TensorRT | 3-5x faster | None |

---

## Troubleshooting

### Common Issues

**Issue:** CUDA out of memory
```python
# Solution: Reduce batch size
model.train(batch=8)  # Instead of 16
```

**Issue:** Low mAP on specific class
```python
# Solution: Add more training data for that class
# Or adjust class weights in loss function
```

**Issue:** Model overfitting
```python
# Solution: Increase augmentation, add dropout, reduce epochs
model.train(
    epochs=50,  # Reduce from 100
    dropout=0.2,  # Add dropout
    degrees=10, translate=0.2  # Increase augmentation
)
```

---

## Pre-trained Datasets

### Public Datasets for Safety Gear

| Dataset | Classes | Images | Link |
|---------|---------|--------|------|
| Safety Helmet Wearing | Helmet/No-Helmet | 7,581 | [Kaggle](https://www.kaggle.com) |
| Hard Hat Workers | Person, Helmet, Vest | 7,000 | [Roboflow](https://roboflow.com) |
| PPE Detection | Multi-class PPE | 3,000 | [GitHub](https://github.com) |
| Construction Safety | Helmet, Vest, Boots | 5,000 | [Academic](https://) |

### Converting Existing Datasets

```python
from ultralytics import YOLO

# Convert COCO format to YOLO
model = YOLO('yolov8n.pt')
model.data = 'coco_ppe.yaml'

# Or use Roboflow for automatic conversion
# Visit: https://roboflow.com
```

---

## Best Practices Summary

✅ **Do:**
- Collect diverse data (different angles, lighting, weather)
- Label consistently across all annotators
- Use at least 500 images per class
- Validate on real-world test data
- Monitor training metrics (loss curves, mAP)

❌ **Don't:**
- Train with too few images (<100 per class)
- Ignore class imbalance
- Use the same data for train and test
- Skip validation before deployment
- Deploy without real-world testing

---

For questions or support, refer to:
- [Ultralytics Documentation](https://docs.ultralytics.com)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- SafeGear AI Issues: [GitHub Issues](https://github.com/yourusername/safear-ai/issues)
