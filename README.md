# 🎤 Binary Speech Emotion Recognition (SER) Project

[![Kaggle](https://img.shields.io/badge/Kaggle-View%20Code-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ammarqurthuby/ser-svm-cnn-biltsm)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format-00629B?style=for-the-badge&logo=ieee&logoColor=white)](IEEE_MTT_S_Conference__IMS_2013_example___1_/main.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-12.2K%20Samples-FF6F00?style=for-the-badge&logo=databricks&logoColor=white)](merged_ser_dataset/)

> **A comprehensive comparative study of SVM, CNN, and Bi-LSTM approaches for binary emotion classification from speech signals**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Dataset](#-dataset)
- [Models Compared](#-models-compared)
- [Quick Start](#-quick-start)
- [Kaggle Notebook](#-kaggle-notebook)
- [Research Paper](#-research-paper)
- [Dependencies](#-dependencies)
- [Authors](#-authors)

---

## 🎯 Overview

This project implements and compares three machine learning approaches for **binary speech emotion recognition**:

- **Traditional ML**: Support Vector Machine (SVM) with RBF kernel
- **Deep Learning (CNN)**: 1D Convolutional Neural Network
- **Deep Learning (RNN)**: Bidirectional Long Short-Term Memory (Bi-LSTM)

### Problem Statement

Classify speech emotions into two categories:
- **Negative** 😠😨😢: angry, disgust, fear, sad
- **Non-Negative** 😊😐😲: happy, neutral, surprise

### Why Binary Classification?

1. **Simplified Decision-Making**: Easier for automated systems to act upon
2. **Higher Accuracy**: Reduced complexity compared to multi-class classification
3. **Clinical Relevance**: Direct application in mental health screening and monitoring

---

## 🏆 Key Results

| Model | Accuracy | F1-Score | Training Time | Best For |
|-------|----------|----------|---------------|----------|
| **1D-CNN** ⭐ | **82.37%** | **0.8658** | 12.5 min | Best overall performance |
| **SVM** | 79.57% | 0.8513 | 3.2 min | Fast training & inference |
| **Bi-LSTM** | 75.67% | 0.8063 | 18.7 min | Benchmark comparison |

### Key Findings

✅ **CNN outperforms Bi-LSTM** despite common assumptions about recurrent networks for sequential data  
✅ **SVM offers best efficiency** for resource-constrained environments  
✅ **Feature representation matters more** than model complexity for this task  

---

## 📊 Dataset

### Sources (4 Datasets Merged)

| Dataset | Samples | Actors | Emotions | Year | Citation |
|---------|---------|--------|----------|------|----------|
| **CREMA-D** | 7,442 | 91 (diverse) | 6 | 2014 | Cao et al. |
| **RAVDESS** | 1,440 | 24 (professional) | 8 | 2018 | Livingstone & Russo |
| **SAVEE** | 480 | 4 (male) | 7 | 2008 | Haq & Jackson |
| **TESS** | 2,800 | 2 (female) | 7 | 2010 | Dupuis & Pichora-Fuller |

### Total Statistics

```
Total Samples: 12,162 audio files (.wav)
├── Negative (63.2%): 7,692 samples
│   ├── Angry:   1,923
│   ├── Disgust: 1,923
│   ├── Fear:    1,923
│   └── Sad:     1,923
│
└── Non-Negative (36.8%): 4,470 samples
    ├── Happy:    1,923
    ├── Neutral:  1,895
    └── Surprise:   652
```

### Dataset Split

- **Training**: 80% (9,729 samples)
- **Testing**: 20% (2,433 samples)
- **Strategy**: Stratified sampling to maintain class balance

---

## 🤖 Models Compared

### 1. Support Vector Machine (SVM)

**Features**: 80D statistical features (MFCC mean + std)

```python
- Kernel: RBF
- Parameters: Grid search optimized (C, gamma)
- Training: ~3.2 minutes
- Accuracy: 79.57%
```

**✅ Best for**: Edge devices, real-time applications, limited resources

---

### 2. 1D Convolutional Neural Network (CNN)

**Features**: 100×40 sequential MFCC features

```python
Architecture:
├── Conv Block 1: 64 filters, kernel=5, BatchNorm, MaxPool
├── Conv Block 2: 128 filters, kernel=5, BatchNorm, MaxPool
├── Conv Block 3: 256 filters, kernel=3, BatchNorm, MaxPool
├── Global Average Pooling
├── Dense: 256 units, Dropout=0.5
└── Output: 2 units (Softmax)

Optimizer: Adam (lr=0.001)
Early Stopping: patience=15
```

**✅ Best for**: High accuracy requirements, moderate computational budget

---

### 3. Bidirectional LSTM (Bi-LSTM)

**Features**: 100×40 sequential MFCC features

```python
Architecture:
├── Bi-LSTM 1: 128 units, return_sequences=True, Dropout=0.3
├── Bi-LSTM 2: 64 units, return_sequences=True, Dropout=0.3
├── Bi-LSTM 3: 32 units, Dropout=0.3
├── Dense: 64 units, Dropout=0.4
└── Output: 2 units (Softmax)

Optimizer: Adam (lr=0.0001)
Early Stopping: patience=20
```

**✅ Best for**: Baseline comparison, research purposes

---

## 🚀 Quick Start

### Option 1: Run on Kaggle (Recommended)

[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/code/ammarqurthuby/ser-svm-cnn-biltsm)

**Advantages:**
- ✅ Pre-configured environment
- ✅ Free GPU access (Tesla P100)
- ✅ Dataset already loaded
- ✅ One-click execution

**Steps:**
1. Click the badge above or visit the link
2. Click **"Copy & Edit"** to create your own version
3. Ensure **GPU is enabled** (Settings → Accelerator → GPU)
4. Click **"Run All"** or execute cells sequentially

---

### Option 2: Run Locally

#### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
scikit-learn
librosa
pandas, numpy, matplotlib, seaborn
```

#### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/binary-ser-project.git
cd binary-ser-project

# Install dependencies
pip install -r requirements.txt

# Download dataset (if not already available)
# Place merged_ser_dataset in the project root
```

#### Run Experiment

```bash
# Open Jupyter Notebook
jupyter notebook ser-svm-cnn-biltsm.ipynb

# Or run via command line (convert notebook to script first)
jupyter nbconvert --to script ser-svm-cnn-biltsm.ipynb
python ser-svm-cnn-biltsm.py
```

#### Expected Runtime

- **Feature Extraction**: ~10-15 minutes (12,162 files)
- **SVM Training**: ~3 minutes
- **CNN Training**: ~12 minutes (with GPU)
- **Bi-LSTM Training**: ~18 minutes (with GPU)
- **Total**: ~45-60 minutes

---

## 📓 Kaggle Notebook

### Live Code & Experiments

🔗 **Official Kaggle Notebook**: [ser-svm-cnn-biltsm](https://www.kaggle.com/code/ammarqurthuby/ser-svm-cnn-biltsm)

### Features

- ✅ **Complete Implementation**: All 3 models with hyperparameter tuning
- ✅ **Dual Feature Extraction**: 
  - Statistical features (80D) for SVM
  - Sequential features (100×40) for CNN/Bi-LSTM
- ✅ **Comprehensive Evaluation**: Accuracy, F1, Confusion Matrix, Training Curves
- ✅ **Reproducible Results**: Fixed random seeds (42)
- ✅ **Well-Documented**: Markdown explanations + inline comments

### Key Sections

1. **Data Loading** - Load 4 datasets from directory structure
2. **Feature Engineering** - Dual extraction strategy (statistical + sequential)
3. **Model Training** - SVM, CNN, Bi-LSTM with early stopping
4. **Evaluation** - Metrics, visualizations, error analysis
5. **Comparison** - Side-by-side performance analysis

---

## 📄 Research Paper

### IEEE Conference Paper (5 Pages)

**Title**: *Binary Speech Emotion Recognition: A Comparative Study of SVM, CNN, and Bi-LSTM Approaches*

**Authors**: Ammar Qurthuby & Habibi  
**Affiliation**: Universitas Syiah Kuala, Department of Informatics Engineering

### Paper Sections

| Section | Content |
|---------|---------|
| **I. Introduction** | Motivation, problem statement, contributions |
| **II. Related Work** | SER approaches, features, binary vs multi-class |
| **III. Methodology** | Dataset (4 sources), features (dual strategy), models (SVM/CNN/Bi-LSTM) |
| **IV. Results** | Performance comparison, confusion matrices, training dynamics |
| **V. Discussion** | Why CNN wins, error analysis, practical implications, limitations |
| **VI. Conclusion** | Summary, recommendations, future work |

### Statistics

- **Pages**: 5 (IEEE 2-column format)
- **Words**: ~3,500
- **Tables**: 3 (Dataset, Performance, Detailed Metrics)
- **References**: 19 peer-reviewed papers (all validated)

### LaTeX Compilation

```bash
cd IEEE_MTT_S_Conference__IMS_2013_example___1_
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use the automated script:
```bash
./compile_and_open.bat  # Windows
```

---

## 📦 Dependencies

### Core Libraries

```txt
# Deep Learning
tensorflow>=2.8.0
keras>=2.8.0

# Traditional ML
scikit-learn>=1.0.2

# Audio Processing
librosa>=0.9.2
soundfile>=0.10.3

# Data Processing
numpy>=1.21.0
pandas>=1.3.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
tqdm>=4.62.0
```

### Installation

```bash
pip install tensorflow scikit-learn librosa pandas numpy matplotlib seaborn tqdm soundfile
```

### Hardware Requirements

**Minimum**:
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 8 GB
- Storage: 5 GB free space

**Recommended**:
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 16 GB
- GPU: NVIDIA GTX 1660 Ti or better (6GB VRAM)
- Storage: 10 GB free space

---

## 👥 Authors

### Ammar Qurthuby
- 🎓 **Program**: Informatics Engineering, Universitas Syiah Kuala
- 📧 **Email**: ammar22@mhs.usk.ac.id
- 🔗 **Kaggle**: [@ammarqurthuby](https://www.kaggle.com/ammarqurthuby)

### Habibi
- 🎓 **Program**: Informatics Engineering, Universitas Syiah Kuala
- 📧 **Email**: habibi123@mhs.usk.ac.id

### Advisor
- **Department**: Informatics Engineering
- **Institution**: Universitas Syiah Kuala, Indonesia

---

## 📚 Citation

If you use this work in your research, please cite:

### BibTeX

```bibtex
@inproceedings{qurthuby2025binary,
  title={Binary Speech Emotion Recognition: A Comparative Study of SVM, CNN, and Bi-LSTM Approaches},
  author={Qurthuby, Ammar and Habibi},
  booktitle={Proceedings of [Conference Name]},
  year={2025},
  organization={Universitas Syiah Kuala}
}
```

### APA

```
Qurthuby, A., & Habibi. (2025). Binary Speech Emotion Recognition: 
A Comparative Study of SVM, CNN, and Bi-LSTM Approaches. 
Proceedings of [Conference Name]. Universitas Syiah Kuala.
```

Check The Code : 
[![Kaggle](https://img.shields.io/badge/Kaggle-View%20Code-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ammarqurthuby/ser-svm-cnn-biltsm)

