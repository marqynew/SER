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
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Kaggle Notebook](#-kaggle-notebook)
- [Research Paper](#-research-paper)
- [Dependencies](#-dependencies)
- [Authors](#-authors)
- [Citation](#-citation)
- [License](#-license)

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

## 📁 Project Structure

```
Klasifikasi-Suara/
│
├── 📄 README.md                          # This file
├── 📄 PAPER_UPDATE_DATASET_4_SOURCES.md  # Latest paper update summary
├── 📄 VALIDASI_REFERENSI_LENGKAP.md      # Reference validation report
│
├── 📓 Notebooks/
│   ├── ser-svm-cnn-biltsm.ipynb          # Main experiment notebook
│   ├── Upload/
│   │   ├── ser-svm-cnn-biltsm.ipynb      # Kaggle version
│   │   └── merged-ser-dataset-creation.ipynb  # Dataset preparation
│   └── (other experiment notebooks...)
│
├── 📂 IEEE_MTT_S_Conference__IMS_2013_example___1_/
│   ├── main.tex                          # LaTeX source
│   ├── main.pdf                          # ✅ Final IEEE paper (5 pages)
│   ├── biblio_ser.bib                    # 19 validated references
│   ├── IEEEtran.cls                      # IEEE template
│   └── IEEEtran.bst                      # IEEE bibliography style
│
├── 📂 merged_ser_dataset/
│   └── merged_ser_dataset/
│       ├── README.md                     # Dataset documentation
│       ├── file_mapping.csv              # File path mappings
│       ├── angry/     (1,923 files)
│       ├── disgust/   (1,923 files)
│       ├── fear/      (1,923 files)
│       ├── sad/       (1,923 files)
│       ├── happy/     (1,923 files)
│       ├── neutral/   (1,895 files)
│       └── surprise/  (652 files)
│
└── 📚 Documentation/
    ├── PAPER_READY_AMMAR_HABIBI.md       # Paper submission guide
    ├── FINAL_CHECKLIST.md                # Pre-submission checklist
    ├── README_INDEX.md                   # Documentation index
    └── (other guides...)
```

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

---

## 📊 Dataset Citations

Please also cite the original dataset creators:

```bibtex
% CREMA-D
@article{cao2014crema,
  title={CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset},
  author={Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur, Ruben C and Nenkova, Ani and Verma, Ragini},
  journal={IEEE Transactions on Affective Computing},
  volume={5}, number={4}, pages={377--390}, year={2014}
}

% RAVDESS
@article{livingstone2018ryerson,
  title={The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)},
  author={Livingstone, Steven R and Russo, Frank A},
  journal={PloS one}, volume={13}, number={5}, pages={e0196391}, year={2018}
}

% SAVEE
@inproceedings{haq2008savee,
  title={Speaker-dependent audio-visual emotion recognition},
  author={Haq, Sanaul and Jackson, Philip JB},
  booktitle={Proceedings of AVSP}, pages={53--58}, year={2008}
}

% TESS
@misc{dupuis2010tess,
  title={Toronto Emotional Speech Set (TESS)},
  author={Dupuis, Kate and Pichora-Fuller, M Kathleen},
  year={2010}, publisher={University of Toronto}
}
```

---

## 🔍 Key Insights

### 1. CNN vs Bi-LSTM Performance Gap

**Why CNN Outperforms?**
- ✅ Better suited for **local pattern recognition** in spectrograms
- ✅ **Built-in dimensionality reduction** through pooling
- ✅ **More stable convergence** during training
- ✅ **Less prone to overfitting** with current feature set

**Why Bi-LSTM Underperforms?**
- ❌ 100-frame sequence may be **insufficient for long-term dependencies**
- ❌ **Overparameterized** for binary classification task
- ❌ Requires **richer temporal features** (prosody, energy contours)

### 2. Feature Engineering Matters

| Feature Type | SVM | CNN/Bi-LSTM |
|--------------|-----|-------------|
| **Statistical** | 80D (mean+std) | ❌ Not suitable |
| **Sequential** | ❌ Not suitable | 100×40 MFCC |
| **Preprocessing** | StandardScaler | Z-score normalization |

### 3. Practical Recommendations

| Scenario | Best Model | Reason |
|----------|-----------|--------|
| **Production Deployment** | 1D-CNN | Best accuracy-efficiency trade-off |
| **Edge Devices / IoT** | SVM | Fast, low memory footprint |
| **Research / Baseline** | All 3 | Comprehensive comparison |
| **Real-time Applications** | SVM | Sub-second inference time |

---

## 🎯 Future Work

### Short-Term (Next 3-6 months)

- [ ] **Enhanced Features**: Add prosodic features (pitch, energy, speaking rate)
- [ ] **Attention Mechanisms**: Implement attention-based Bi-LSTM
- [ ] **Ensemble Methods**: Combine SVM + CNN predictions
- [ ] **Data Augmentation**: Pitch shifting, time stretching, noise injection

### Medium-Term (6-12 months)

- [ ] **Cross-Dataset Evaluation**: Test generalization across CREMA-D, RAVDESS, SAVEE, TESS individually
- [ ] **Multi-Task Learning**: Joint emotion + speaker recognition
- [ ] **Transfer Learning**: Fine-tune pre-trained models (VGGish, wav2vec2)
- [ ] **Real-Time System**: Build web API for live emotion detection

### Long-Term (1+ years)

- [ ] **Clinical Deployment**: Mental health monitoring application
- [ ] **Multilingual Extension**: Support for non-English languages
- [ ] **Explainable AI**: Visualize model attention on audio segments
- [ ] **Mobile App**: Android/iOS app for on-device inference

---

## 🐛 Known Issues & Limitations

### Current Limitations

1. **Class Imbalance**: 63.2% Negative vs 36.8% Non-Negative
   - **Mitigation**: Stratified splitting, balanced class weights
   
2. **Surprise Emotion**: Only 652 samples (5.4%)
   - **Impact**: Frequent misclassification due to high arousal
   
3. **Dataset Variability**: 4 datasets with different recording conditions
   - **Impact**: Cross-dataset generalization not fully tested
   
4. **Feature Representation**: 100-frame limit may truncate long utterances
   - **Impact**: Loss of temporal information for Bi-LSTM

### Troubleshooting

**Issue**: Out of Memory during feature extraction
- **Solution**: Process in batches (500 files per batch)

**Issue**: GPU not detected in TensorFlow
- **Solution**: Install CUDA 11.2 + cuDNN 8.1

**Issue**: Librosa installation fails
- **Solution**: `pip install llvmlite --upgrade` then `pip install librosa`

---

## 📜 License

This project is licensed under the **MIT License** - see below for details:

```
MIT License

Copyright (c) 2025 Ammar Qurthuby & Habibi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Dataset Licenses

- **CREMA-D**: Available for research purposes (cite original paper)
- **RAVDESS**: Creative Commons Attribution-ShareAlike 4.0
- **SAVEE**: Available for research purposes (cite original paper)
- **TESS**: Available for research purposes (cite original paper)

---

## 🙏 Acknowledgments

Special thanks to:

- **Dataset Creators**: Cao et al. (CREMA-D), Livingstone & Russo (RAVDESS), Haq & Jackson (SAVEE), Dupuis & Pichora-Fuller (TESS)
- **Universitas Syiah Kuala**: Department of Informatics Engineering for computational resources and guidance
- **Kaggle Community**: For providing free GPU resources and platform for sharing research
- **Open Source Libraries**: TensorFlow, scikit-learn, librosa, and all contributors

---

## 📞 Contact & Support

### Questions or Issues?

- 📧 **Email**: ammar22@mhs.usk.ac.id
- 💬 **Kaggle Discussion**: Comment on the [notebook](https://www.kaggle.com/code/ammarqurthuby/ser-svm-cnn-biltsm)
- 🐛 **Bug Reports**: Open an issue on GitHub (if repository is public)

### Stay Updated

- ⭐ **Star this repo** to get notified of updates
- 👀 **Watch** for new releases and improvements
- 🍴 **Fork** to create your own experiments

---

## 📈 Project Statistics

![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/binary-ser-project)
![GitHub code size](https://img.shields.io/github/languages/code-size/yourusername/binary-ser-project)
![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/binary-ser-project)

---

<div align="center">

### 🌟 If this project helped you, please consider giving it a star! 🌟

**Built with by Ammar Qurthuby & Habibi**

*Universitas Syiah Kuala - Informatics Engineering*

[📄 Paper](IEEE_MTT_S_Conference__IMS_2013_example___1_/main.pdf) • 
[💻 Code](https://www.kaggle.com/code/ammarqurthuby/ser-svm-cnn-biltsm) • 
[📊 Dataset](merged_ser_dataset/) • 
[📚 Docs](PAPER_READY_AMMAR_HABIBI.md)

</div>

---

**Last Updated**: December 26, 2025  
**Version**: 1.0.0  
