# 🃏 Playing Card Detection System — YOLOv8

A real-time playing card detection system built using **YOLOv8** deep learning architecture. The model is trained to detect and classify all **52 standard playing cards** through a live webcam feed, deployed entirely in the browser using **ONNX Runtime**.

> 👥 Group Members: 73, 93, 33, 65 | 📚 Subject: Machine Learning | 👨‍🏫 Instructor: Sir Hamza Farooqi

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Technology Stack](#️-technology-stack)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Configuration](#️-training-configuration)
- [Results](#-results)
- [System Architecture](#-system-architecture)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)

---

## 📌 Project Overview

This system uses **YOLOv8 Small (yolov8s)** to detect playing cards in real time. The trained model is exported to **ONNX format** and deployed through a browser-based HTML interface — no installation required.

### 🎯 Objectives
- Train a deep learning model to detect all 52 playing cards
- Achieve real-time detection performance (>15 FPS)
- Deploy through a browser-based interface using ONNX Runtime
- Run entirely locally — no cloud or API calls needed

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Training Platform | Google Colab (Tesla T4 GPU) |
| Detection Framework | Ultralytics YOLOv8 |
| Language | Python 3.10+ |
| Deployment Runtime | ONNX Runtime Web |
| Interface | HTML5, JavaScript, Canvas API |
| Dataset Management | Roboflow |
| Model Format | ONNX (opset 12) |

---

## 📂 Dataset

- **Source:** Roboflow Community (`poker-eqo1i / playing-card-dedbm`)
- **Total Classes:** 52 (all standard playing cards)
- **Image Size:** Resized to 640×640 during training

### Card Classes:
| Suit | Cards |
|---|---|
| ♣ Clubs | AC, 2C, 3C, 4C, 5C, 6C, 7C, 8C, 9C, 10C, JC, QC, KC |
| ♦ Diamonds | AD, 2D, 3D, 4D, 5D, 6D, 7D, 8D, 9D, 10D, JD, QD, KD |
| ♥ Hearts | AH, 2H, 3H, 4H, 5H, 6H, 7H, 8H, 9H, 10H, JH, QH, KH |
| ♠ Spades | AS, 2S, 3S, 4S, 5S, 6S, 7S, 8S, 9S, 10S, JS, QS, KS |

### Data Split:
- Training: ~70%
- Validation: ~20%
- Test: ~10%

### Data Augmentation Applied:
- Horizontal flipping (50% probability)
- Rotation (±10 degrees)
- Translation (±10%), Scale variation
- HSV color jittering (hue, saturation, value)
- Mosaic augmentation
- Mixup augmentation (10% probability)

---

## 🤖 Model Architecture

- **Model:** YOLOv8 Small (`yolov8s.pt`)
- **Parameters:** ~11 million
- **Input Shape:** `(1, 3, 640, 640)`
- **Output Shape:** `(1, 56, 8400)`
- **Backbone:** CSPDarknet with C2f blocks
- **Detection Head:** Anchor-free Detect head (52 classes)

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 200 (Early stopped at 192) |
| Batch Size | 16 |
| Image Size | 640×640 |
| Optimizer | AdamW |
| Initial LR | 0.001 |
| Final LR | 0.00001 |
| Weight Decay | 0.0005 |
| Warmup Epochs | 3 |
| Patience (Early Stop) | 30 |
| Box Loss Gain | 7.5 |
| Class Loss Gain | 0.5 |
| DFL Loss Gain | 1.5 |

---

## 📊 Results

### Final Validation Metrics (Best Model — Epoch 162):

| Metric | Value |
|---|---|
| **mAP50** | **99.4%** |
| **mAP50-95** | **90.1%** |
| Precision | 96.9% |
| Recall | 98.0% |

### Training Progress Highlights:

| Epoch | mAP50 | mAP50-95 |
|---|---|---|
| 1 | 1.05% | 0.68% |
| 20 | 68.4% | 47.5% |
| 50 | 98.5% | 78.3% |
| 100 | 99.1% | 86.5% |
| 162 (Best) | **99.4%** | **90.0%** |

> ✅ Training stopped early at epoch 192 — best model saved at **epoch 162**.

### Inference Speed:
- Preprocessing: 0.4ms
- Inference: 4.8ms
- Postprocess: 0.0ms
- **Total: ~5.2ms per image**

---

## 🏗️ System Architecture

### Training Pipeline:
```
Roboflow Dataset → Google Colab → YOLOv8 Training
→ PyTorch Model (.pt) → ONNX Export (.onnx) → Download
```

### Deployment Pipeline:
```
User Browser → Load ONNX Model → Access Webcam
→ Capture Frame → Preprocess (640×640, normalize)
→ Model Inference → NMS Post-processing
→ Draw Bounding Boxes → Display on Canvas
```

---

## 🚀 How to Run

### Step 1 — Train the Model (Google Colab)
```bash
# Open the notebook in Google Colab
# Run all cells to train the model
# Download: runs/detect/card_detector_v2/weights/best.onnx
```

### Step 2 — Run the Web Interface
1. Save `card_detection.html` to your local machine
2. Open the file in **Chrome or Edge** (WebGL required)
3. Upload the `best.onnx` model file
4. Allow webcam access
5. Click **Start Detection**

### Step 3 — Start Detecting!
- Hold any playing card in front of your webcam
- The system will detect and label it in real time with a bounding box and confidence score

---

## 📁 Project Structure

```
Playing-Card-Detection-YOLOv8/
│
├── playing_card_detection_yolov8_training.ipynb  # Training notebook (Google Colab)
├── card_detection.html                           # Browser-based detection interface
├── best.onnx                                     # Exported trained model
├── Playing_Card_Detection_Project_Report.pdf     # Full project report
└── README.md                                     # Project documentation
```

---

## 🌍 Real-World Applications

- **Casino Monitoring** — Automated card tracking and fraud detection
- **Online Gaming** — AR card games and virtual poker
- **Education** — Interactive learning tools for card games
- **Accessibility** — Assistive technology for visually impaired players
- **Card Counting Prevention** — Real-time detection in casinos

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| Model won't load | Check ONNX export settings, use opset=12 |
| Low FPS | Use lighter model (yolov8n) or reduce image size |
| Poor accuracy | Retrain with more epochs or larger model |
| Camera not working | Check browser permissions in Chrome/Edge |

---

## 👤 Author

**Hamza Asif**
