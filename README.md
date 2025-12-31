# 🎥 Real-Time Object Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00DFA2?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-Latest-red?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**A lightweight real-time object detection system using YOLOv8 and webcam**

Simple • Fast • Efficient

</div>

---

## 📖 About

This is a minimal implementation of real-time object detection using YOLOv8 and OpenCV. The project captures live video from your webcam and detects objects in real-time with bounding boxes and confidence scores.

Perfect for learning computer vision basics or as a starting point for more complex projects!

---

## ✨ Features

- 🎯 Real-time object detection using YOLOv8n (nano model)
- 📹 Webcam integration with OpenCV
- 🟢 Green bounding boxes with class labels
- 💯 Confidence score display (threshold: 0.5)
- ⌨️ Simple keyboard control (press 'q' to quit)
- 🚀 Lightweight and fast

---

## 🛠️ Requirements

```bash
Python >= 3.8
ultralytics
opencv-python
```

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/pragambesh-moro/detection_proj.git
cd detection_proj
```

2. **Install dependencies**
```bash
pip install ultralytics opencv-python
```

3. **Run the script**
```bash
python detection.py
```

That's it! The YOLOv8n model will be automatically downloaded on first run.

---

## 🚀 Usage

Simply run the script and point your webcam at objects:

```bash
python detection.py
```

**Controls:**
- Press **'q'** to quit the application
- The detection window shows real-time results with bounding boxes

**What it detects:**
YOLOv8n can detect 80 different object classes including:
- People
- Vehicles (cars, trucks, bikes)
- Animals (cats, dogs, birds)
- Common objects (phones, laptops, cups, etc.)

---

## 📝 How It Works

The script follows a simple workflow:

1. **Initialize Model**: Loads the YOLOv8n (nano) pre-trained model
2. **Start Webcam**: Opens your default camera (index 0)
3. **Capture Frames**: Continuously reads frames from the camera
4. **Detect Objects**: Runs YOLOv8 detection on each frame
5. **Draw Boxes**: Adds green bounding boxes and labels for objects with >50% confidence
6. **Display**: Shows the annotated frame in a window
7. **Loop**: Repeats until you press 'q'

---

## 🔧 Customization

You can easily modify the script to suit your needs:

### Change Detection Threshold
```python
if confidence > 0.5:  # Change 0.5 to your desired threshold (0.0 to 1.0)
```

### Use Different Camera
```python
cam_cap = cv2.VideoCapture(0)  # Change 0 to 1, 2, etc. for other cameras
```

### Change Bounding Box Color
```python
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # (B, G, R) format
# Example: (0, 0, 255) for red, (255, 0, 0) for blue
```

### Use a Different YOLOv8 Model
```python
model = YOLO('yolov8n.pt')  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
```

Model size comparison:
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (slowest, most accurate)

---

## 📊 Performance

**System Requirements:**
- Webcam or camera device
- ~6MB for YOLOv8n model
- Runs smoothly on CPU (GPU optional)

**Expected Performance:**
- CPU: 10-30 FPS (depending on your processor)
- GPU: 60+ FPS

---

## 🐛 Troubleshooting

### Camera not opening?
- Check if another application is using the camera
- Try changing the camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`
- On Linux, you might need permissions to access the camera

### Model not downloading?
- Check your internet connection
- The model will be saved to your local ultralytics cache after first download

### Low FPS?
- Try using a smaller input resolution
- Consider using GPU acceleration with CUDA
- Use the yolov8n (nano) model for better performance

---

## 📚 Learn More

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Python Tutorial](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [YOLOv8 GitHub Repository](https://github.com/ultralytics/ultralytics)

---

## 🤝 Contributing

Feel free to fork this project and make improvements! Some ideas:
- Add support for video file input
- Implement object tracking
- Save detection results to a file
- Add a GUI for easier control
- Support for custom trained models

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Pragambesh Moro**

- GitHub: [@pragambesh-moro](https://github.com/pragambesh-moro)
- Project: [detection_proj](https://github.com/pragambesh-moro/detection_proj)

---

<div align="center">

**⭐ If you find this helpful, consider giving it a star!**

Built with 🎯 YOLOv8 and 💚 OpenCV

</div>
