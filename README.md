# 🔍 Detection Project

<div align="center">

![Detection Banner](https://img.shields.io/badge/Computer_Vision-Detection-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

**A modern object detection system powered by deep learning**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

This project implements a state-of-the-art object detection system capable of identifying and localizing objects in images and video streams. Built with modern deep learning frameworks, it provides both accuracy and real-time performance for various detection tasks.

### Key Highlights

- 🎯 High-accuracy object detection
- ⚡ Real-time inference capabilities
- 🔧 Easy-to-use API
- 📊 Comprehensive evaluation metrics
- 🎨 Visualization tools included

---

## ✨ Features

- **Multiple Detection Models**: Support for YOLO, Faster R-CNN, and custom architectures
- **Pre-trained Weights**: Quick start with pre-trained models on common datasets
- **Custom Training**: Train on your own dataset with minimal configuration
- **Batch Processing**: Efficient processing of multiple images
- **Video Detection**: Real-time detection on video streams
- **Export Options**: Model export for deployment (ONNX, TensorRT)

---

## 🚀 Quick Start

### Prerequisites

```bash
Python >= 3.8
CUDA >= 11.0 (for GPU support)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/pragambesh-moro/detection_proj.git
cd detection_proj

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from detector import ObjectDetector

# Initialize detector
detector = ObjectDetector(model='yolov5', weights='pretrained')

# Detect objects in an image
results = detector.detect('path/to/image.jpg')

# Visualize results
detector.visualize(results, save_path='output.jpg')
```

---

## 📁 Project Structure

```
detection_proj/
├── data/                  # Dataset directory
│   ├── train/
│   ├── val/
│   └── test/
├── models/                # Model architectures
│   ├── yolo.py
│   ├── faster_rcnn.py
│   └── custom_model.py
├── configs/               # Configuration files
│   └── config.yaml
├── utils/                 # Utility functions
│   ├── preprocessing.py
│   ├── postprocessing.py
│   └── visualization.py
├── weights/               # Trained model weights
├── scripts/               # Training and evaluation scripts
│   ├── train.py
│   └── evaluate.py
├── notebooks/             # Jupyter notebooks for demos
├── requirements.txt
└── README.md
```

---

## 🎓 Training

### Prepare Your Dataset

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

### Train the Model

```bash
python scripts/train.py \
    --config configs/config.yaml \
    --data data/ \
    --epochs 100 \
    --batch-size 16
```

### Configuration Options

Edit `configs/config.yaml` to customize:
- Model architecture
- Hyperparameters
- Data augmentation
- Training settings

---

## 📊 Evaluation

Run evaluation on the test set:

```bash
python scripts/evaluate.py \
    --weights weights/best.pt \
    --data data/test/
```

### Metrics

The evaluation provides:
- Mean Average Precision (mAP)
- Precision and Recall curves
- Confusion matrix
- Inference time benchmarks

---

## 🎯 Inference Examples

### Single Image Detection

```python
import cv2
from detector import ObjectDetector

# Load detector
detector = ObjectDetector(model='yolov5', weights='weights/best.pt')

# Run detection
image = cv2.imread('sample.jpg')
results = detector.detect(image, conf_threshold=0.5)

# Print results
for obj in results:
    print(f"Class: {obj['class']}, Confidence: {obj['confidence']:.2f}")
```

### Video Stream Detection

```python
import cv2
from detector import ObjectDetector

detector = ObjectDetector(model='yolov5', weights='weights/best.pt')
cap = cv2.VideoCapture(0)  # Use webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = detector.detect(frame, conf_threshold=0.5)
    annotated_frame = detector.draw_boxes(frame, results)
    
    cv2.imshow('Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🛠️ Advanced Features

### Model Export

Export trained models for deployment:

```bash
# Export to ONNX
python scripts/export.py \
    --weights weights/best.pt \
    --format onnx

# Export to TensorRT
python scripts/export.py \
    --weights weights/best.pt \
    --format tensorrt
```

### Batch Processing

Process multiple images efficiently:

```python
from detector import BatchDetector

batch_detector = BatchDetector(model='yolov5', weights='weights/best.pt')
results = batch_detector.detect_batch('path/to/images/', batch_size=8)
```

---

## 📈 Performance

| Model | mAP@0.5 | FPS (GPU) | Parameters |
|-------|---------|-----------|------------|
| YOLOv5s | 0.87 | 140 | 7.2M |
| YOLOv5m | 0.91 | 90 | 21.2M |
| YOLOv5l | 0.94 | 60 | 46.5M |

*Benchmarked on NVIDIA RTX 3090*

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- YOLOv5 by Ultralytics
- PyTorch team for the deep learning framework
- OpenCV for computer vision utilities
- The open-source community

---

## 📮 Contact

**Pragambesh Moro**

- GitHub: [@pragambesh-moro](https://github.com/pragambesh-moro)
- Project Link: [https://github.com/pragambesh-moro/detection_proj](https://github.com/pragambesh-moro/detection_proj)

---

## 🔄 Changelog

### v1.0.0 (Current)
- Initial release
- Support for YOLOv5 models
- Basic training and inference pipelines
- Video stream detection
- Model export functionality

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by Pragambesh Moro

</div>
