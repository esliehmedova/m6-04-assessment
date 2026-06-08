# 🐱 Cat Detection with YOLOv8 (yolo26n)

A fine-tuned object detection model that identifies cats in images using the YOLO26 nano architecture. Trained on a cleaned dataset of 3,327 images, achieving **93.4% mAP@0.5** on the test set.

---

## 📊 Results

| Metric | Value |
|---|---|
| mAP@0.5 | **0.9342** |
| mAP@0.5:0.95 | 0.7490 |
| Precision | 0.9418 |
| Recall | 0.8591 |

> Trained for 30 epochs on a Colab T4 GPU (~1.5 hours). Val mAP continued climbing through the final epochs with no signs of overfitting.

---

## 🗂️ Dataset

- **3,327** image-label pairs (single class: `cat`)
- Raw download contained ~1,000 duplicate files (`(N).jpg` style) which were cleaned to prevent data leakage across splits
- Image sizes vary widely (640×640 up to ~6000×7500); YOLO resizes all inputs to 640×640 internally
- Split into train / val / test sets

---

## 🧠 Model

**Architecture:** `yolo26n` (YOLO26 Nano — 2.4M parameters)

Chosen for three reasons:
1. **Task fits the model** — COCO-pretrained YOLO26 already knows cats from 80-class pretraining; this is fine-tuning, not training from scratch
2. **Speed** — Trains in ~1.5h on Colab T4; larger variants (e.g. `yolo26s`) would double training time with marginal gain on a single-class task
3. **Deployability** — ~5 MB checkpoint, trivial to export as ONNX for serving

---

## 🛠️ Tech Stack

- Python
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- PyTorch
- Google Colab (T4 GPU)

---

## 🚀 Usage

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.predict("your_image.jpg")
results[0].show()
```

---

## 📁 Project Structure

```
cat-detection/
├── data.yaml          # Dataset config (paths, class names)
├── yolo26n.pt         # Trained model weights
├── notebook.ipynb     # Full training & evaluation notebook
└── README.md
```

---

## 💡 Potential Improvements

- Try `imgsz=960` to preserve detail in high-resolution images
- Stronger augmentation pipeline
- Experiment with `yolo26s` for marginal precision gains
