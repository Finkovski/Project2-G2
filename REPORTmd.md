# 📄 Report - Project 2 CIFAR-10 Image Classification

## ✨ Project Goal
Build, compare, and analyze different deep learning models to classify images from the CIFAR-10 dataset into 10 predefined categories.

---

## 🔢 Data Preprocessing
- **Dataset:** CIFAR-10 (60,000 color images, 10 classes)
- **Normalization:** Pixel values scaled to [0, 1]
- **Resizing:** Images resized to 160x160 for MobileNetV2
- **Data pipeline:** tf.data API used for performance optimization

---

## 🧠 Model Architectures

### 1. Custom CNN (from scratch)
- Layers:
  - 2 Convolutional + MaxPooling layers
  - Flatten layer
  - Dense layer with Dropout
- Performance:
  - Test Accuracy: ~70%-75%

### 2. Transfer Learning (MobileNetV2)
- Base Model: MobileNetV2 pretrained on ImageNet
- Head: GlobalAveragePooling + Dense(128) + Dropout(0.3) + Dense(10)
- Strategy:
  - Freeze base initially
  - Fine-tune top 50 layers
- Performance:
  - Test Accuracy: ~90%-91%

---

## 🔁 Training Details
- Optimizer: Adam
- Batch Size: 32 for MobileNetV2, 64 for Custom CNN
- Epochs: 10 + fine-tuning (MobileNetV2), 20 (CNN)
- EarlyStopping used for stabilization

---

## 📈 Evaluation Metrics
- Accuracy score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

---

## 🏆 Best Model
- **MobileNetV2** (Transfer Learning)
- Fine-tuning significantly improved results
- Robust against overfitting

---

## 📘 Files Overview
- `Project2_G2.ipynb`: Main final model
- `Project2_G2_other_models.ipynb`: Additional experiments
- `requirements.txt`: Package dependencies
- `REPORTmd.md` and `REPORTpdf.pdf`: Project report

---

# ✨ End of Report