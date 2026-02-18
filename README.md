# 🧠 Brain Stroke CT Multi-Class Classification using ResNet18

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

## 📌 Project Overview
This project provides a Deep Learning-based diagnostic tool to automate the classification of Brain CT scans. Utilizing the **ResNet18** architecture, the model categorizes images into three distinct classes:
* **Normal**: Healthy brain scans.
* **Bleeding (Hemorrhagic)**: Identification of intracranial hemorrhage.
* **Ischemia (Ischemic)**: Identification of restricted blood flow/clots.

The system is designed for high-performance medical imaging, utilizing **Memory Mapping (memmap)** to handle large datasets efficiently without exhausting system RAM.



---

## 🚀 Key Features
* **Architecture:** ResNet18 with Transfer Learning (Pre-trained on ImageNet).
* **Efficiency:** NumPy `memmap` integration for lightning-fast data I/O.
* **Explainability:** (Optional) Support for Grad-CAM to visualize stroke regions.
* **Interactive CLI:** A dedicated inference script for real-time diagnostic testing.
* **Robustness:** Stratified dataset splitting and dynamic data augmentation.

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   git clone https://github.com/Balasubramanyarajan/Brain-Stroke-CT
   cd Brain-Stroke-CT
2. **Install Dependencies:**
   pip install -r requirements.txt
3. Dataset Configuration: Place your dataset in the directory specified in the code. The script expects folders named Normal, Bleeding, and Ischemia.

**Interactive Prediction (Inference)**
Test the model on any new image via the interactive terminal:
  python predict.py
How it works:
    The script loads the trained weights.
    It prompts you for an image path.
    You can drag and drop an image file into the terminal.
    The system outputs the Class Label, Confidence Score, and displays the scan with the result.
    
📊 Evaluation
The model is evaluated using a Confusion Matrix and a Classification Report (Precision, Recall, and F1-Score).
Technical Solutions:
    ->Handling Large Data: Used np.lib.format.open_memmap to process thousands of images without loading them all into RAM at once.
    ->Class Imbalance: Implemented Stratified Shuffling to ensure the model learns features from all stroke types equally.
    ->Optimization: Used AdamW with CosineAnnealingLR to ensure smooth convergence and prevent overfitting. 

**📂 Project Structure**
├── ct_memmap_cache/          # Auto-generated directory
│   ├── best_resnet18.pt      # Trained model weights (Best Accuracy)
│   ├── labels_int64.npy      # Memmap cache for labels
│   ├── images_224x224_uint8.npy # Memmap cache for preprocessed images
│   └── meta.json             # Dataset metadata (class mapping, etc.)
│
├── brain_stroke_classifier.py # Main training & preprocessing script (.py)
├── predict.py                # Interactive CLI tool for single image inference
├── requirements.txt          # List of Python dependencies
├── .gitignore                # Rules to exclude large data/cache from Git
├── LICENSE                   # MIT License file
└── README.md                 # Project documentation and results
