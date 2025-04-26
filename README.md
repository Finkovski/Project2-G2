
# 🚀 CIFAR-10 Image Classification - Project 2

## ✨ Project Overview
This project focuses on image classification using the CIFAR-10 dataset.  
We trained and compared two models:
- A **Custom Convolutional Neural Network (CNN)**
- A **Transfer Learning model** using **MobileNetV2** pretrained on ImageNet.

Both models are evaluated thoroughly and visualized.

---

## 🗂 Project Structure
```
Project2_G2_complete.ipynb       # Main Notebook (training, evaluation)
REPORT.pdf                       # Full report explaining methods & results
REPORT.md                        # Report in Markdown
PPT Presentation                 # PowerPoint for presentation
requirements.txt                 # Required libraries
Flask/Gradio App (optional)       # Model deployment
```

---

## 📦 Setup and Installation

1. Clone the repository or download the ZIP.
2. Install requirements:
```bash
pip install -r requirements.txt
```
3. Open `Project2_G2_complete.ipynb` in Jupyter or Google Colab.

---

## 🧠 Models Used

### 1. Custom CNN
- Conv2D + MaxPooling layers
- Dense(128) + Dropout
- Final Softmax output layer

### 2. Transfer Learning (MobileNetV2)
- Frozen base model initially
- Custom Dense layers
- Fine-tuned top layers for better performance

---

## 📊 Evaluation Metrics
- **Accuracy**
- **Precision, Recall, F1-Score** (for Custom CNN)
- **Confusion Matrix** (for Custom CNN)

---

## 🏆 Final Results
- Custom CNN Accuracy: ~70-75%
- MobileNetV2 Accuracy: ~90-91%

---

## 🌍 Deployment
- Model deployed using **Gradio** / (optional) Flask for predictions on user-uploaded images.

---

## 📘 Future Improvements
- Add data augmentation
- Try other pretrained architectures like ResNet50 or EfficientNet
- Use TensorFlow Serving for production deployment

---

# ✨ Thank you!
