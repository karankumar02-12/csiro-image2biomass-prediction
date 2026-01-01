# 🌾 CSIRO Image2Biomass Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Active-20BEFF.svg)](https://www.kaggle.com/competitions/csiro-biomass)
[![Rank](https://img.shields.io/badge/Rank-2090%2F2795-success.svg)]()

> Predicting pasture biomass from images using ensemble machine learning to help farmers optimize livestock grazing decisions.

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Key Features](#-key-features)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Future Improvements](#-future-improvements)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Problem Statement

Farmers need accurate pasture biomass measurements to make informed grazing decisions. Traditional manual sampling is time-consuming and expensive. This competition challenges participants to build models that predict five types of biomass measurements from pasture images:

- **Dry_Green_g**: Dry weight of green vegetation (weight: 0.1)
- **Dry_Dead_g**: Dry weight of dead vegetation (weight: 0.1)
- **Dry_Clover_g**: Dry weight of clover (weight: 0.1)
- **GDM_g**: Green dry matter (weight: 0.2)
- **Dry_Total_g**: Total dry biomass (weight: 0.5)

**Evaluation Metric**: Weighted R² score computed globally across all predictions, with higher weights on total biomass.

---

## 🚀 Solution Overview

This solution employs a **multi-model ensemble approach** combining computer vision feature engineering with gradient boosting algorithms to predict biomass from pasture images.

### Architecture

```
Image Input (128x128)
    ↓
Feature Extraction (48 features)
├── RGB Statistics (21 features)
├── HSV Color Space (6 features)
├── LAB Color Space (6 features)
├── NDVI Vegetation Index (5 features)
├── Green Ratio (3 features)
└── Texture Features (7 features)
    ↓
Multi-Target Encoding
    ↓
5-Model Ensemble
├── LightGBM (Primary)
├── XGBoost
├── Random Forest
├── Gradient Boosting
└── LightGBM (Variant)
    ↓
Weighted Averaging
    ↓
Final Predictions
```

---

## ✨ Key Features

### 1. **Comprehensive Feature Engineering**
- **48 hand-crafted features** extracted from each image
- Multi-scale color space analysis (RGB, HSV, LAB)
- **NDVI (Normalized Difference Vegetation Index)** for vegetation health
- Texture and gradient features for structural information

### 2. **Robust Cross-Validation**
- **GroupKFold (5 splits)** ensuring no image leakage between folds
- Group-based splitting by image ID prevents overfitting

### 3. **Intelligent Ensemble Strategy**
- **5 diverse models** with complementary strengths
- **Inverse RMSE weighting** giving more influence to better performers
- Hyperparameter tuning for each model variant

### 4. **Seed Variation Strategy**
- Random seed variation (123, 456, 789, 999) for submission diversity
- Reduces variance and improves generalization

---

## 📊 Results

### Cross-Validation Performance

| Model | CV RMSE | Weight in Ensemble |
|-------|---------|-------------------|
| **XGBoost** | 17.6503 | 0.212 |
| **LightGBM-2** | 17.8744 | 0.209 |
| **LightGBM** | 18.0528 | 0.207 |
| **Gradient Boosting** | 17.9334 | 0.208 |
| **Random Forest** | 19.2901 | 0.194 |
| **Ensemble** | **~17.5** | — |

### Competition Standing
- **Rank**: 2090 / 2795 (Top 75%)
- **Metric**: Weighted R² score
- **Approach**: Traditional ML with feature engineering

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM recommended
- GPU optional (CPU sufficient for inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/csiro-image2biomass-prediction.git
cd csiro-image2biomass-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data
```bash
# Using Kaggle API
kaggle competitions download -c csiro-biomass
unzip csiro-biomass.zip -d data/
```

---

## 💻 Usage

### Quick Start - Run Full Pipeline

```bash
# Run the complete solution notebook
jupyter notebook notebooks/03_final_solution.ipynb
```

### Using Modular Code

```python
from src.feature_extraction import extract_image_features
from src.models import train_ensemble_models
from src.config import Config

# Extract features
features = extract_image_features('path/to/image.jpg')

# Train models
models, predictions = train_ensemble_models(
    X_train, y_train, 
    config=Config()
)
```

### Generate Predictions

```python
# Load test data
import pandas as pd
from src.utils import generate_submission

test = pd.read_csv('data/test.csv')

# Generate predictions
submission = generate_submission(models, test)
submission.to_csv('submission.csv', index=False)
```

---

## 📁 Project Structure

```
csiro-image2biomass-prediction/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── .gitignore                        # Git ignore rules
│
├── notebooks/
│   ├── 01_eda_exploration.ipynb      # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Feature extraction experiments
│   └── 03_final_solution.ipynb       # Competition solution
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration parameters
│   ├── feature_extraction.py        # Image feature functions
│   ├── models.py                     # Model training pipeline
│   └── utils.py                      # Helper utilities
│
├── docs/
│   ├── methodology.md                # Detailed approach
│   ├── results.md                    # Performance analysis
│   └── images/                       # Visualizations
│
├── data/
│   └── README.md                     # Data download instructions
│
└── experiments/
    └── seed_experiments.md           # Random seed results
```

---

## 🔬 Methodology

### Feature Engineering Rationale

#### 1. **RGB Statistics (21 features)**
Basic color distribution capturing overall image appearance and lighting conditions.

#### 2. **HSV Color Space (6 features)**
Separates color (Hue) from intensity, more robust to lighting variations in field conditions.

#### 3. **LAB Color Space (6 features)**
Perceptually uniform color space, useful for distinguishing green vegetation from dead matter.

#### 4. **NDVI - Vegetation Index (5 features)**
```
NDVI = (Green - Red) / (Green + Red)
```
Standard remote sensing metric for vegetation health and density. Higher NDVI indicates healthier, denser vegetation.

#### 5. **Green Ratio (3 features)**
```
Green Ratio = Green / (Blue + Green + Red)
```
Simple but effective metric for green vegetation presence.

#### 6. **Texture Features (7 features)**
Gradient and standard deviation features capture spatial patterns and texture complexity.

### Model Selection

**Why Gradient Boosting Ensemble?**
- Handles non-linear relationships between image features and biomass
- Robust to outliers in field data
- Efficient with tabular features
- Ensemble reduces variance and improves generalization

**Why Not Deep Learning?**
- Limited training data (357 images)
- Hand-crafted features provide strong baseline
- Faster training and inference
- More interpretable for agricultural applications

---

## 🎓 Key Learnings

1. **Domain-specific features matter**: NDVI and green ratio significantly outperformed raw pixel values
2. **Group-based CV is critical**: Prevents overfitting when multiple targets come from same image
3. **Ensemble diversity**: Combining tree-based models with different hyperparameters improves robustness
4. **Seed variation**: Multiple submissions with different seeds helps find optimal generalization

---

## 🚀 Future Improvements

### Short-term Enhancements
- [ ] Add spatial features (image segmentation)
- [ ] Incorporate weather/location metadata if available
- [ ] Experiment with feature selection (remove redundant features)
- [ ] Try stacking with meta-learner

### Long-term Directions
- [ ] **Transfer Learning**: Fine-tune EfficientNet or ResNet on pasture images
- [ ] **Multi-task Learning**: Joint prediction of all targets with shared representations
- [ ] **Augmentation**: Rotation, flip, color jittering to increase effective dataset size
- [ ] **Attention Mechanisms**: Focus on relevant image regions (green vs dead vegetation)
- [ ] **Semi-supervised Learning**: Leverage unlabeled pasture images

---

## 📚 References

- [NDVI in Agriculture](https://gisgeography.com/ndvi-normalized-difference-vegetation-index/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Color Space Conversions](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html)

---

## 🙏 Acknowledgments

- **CSIRO** for organizing the competition and providing the dataset
- **Kaggle Community** for discussions and insights
- **Competition Timeline**: Oct 28, 2025 - Jan 28, 2026

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

⭐ **If you found this helpful, please consider giving it a star!**
