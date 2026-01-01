# ----- docs/methodology.md -----
# Methodology: CSIRO Image2Biomass Prediction

## Overview
This document provides a detailed explanation of the methodology used to predict pasture biomass from images.

## 1. Problem Formulation

### Multi-Target Regression
Each image has 5 associated biomass measurements:
- Dry_Green_g (green vegetation)
- Dry_Dead_g (dead vegetation)
- Dry_Clover_g (clover specifically)
- GDM_g (green dry matter)
- Dry_Total_g (total biomass)

We treat this as 5 separate regression problems with shared image features.

### Weighted Evaluation
The competition uses weighted R²:
```
R² = 1 - (Σ w_i * (y_i - ŷ_i)²) / (Σ w_i * (y_i - ȳ)²)
```
Where weights are: [0.1, 0.1, 0.1, 0.2, 0.5] for the 5 targets respectively.

## 2. Feature Engineering

### 2.1 RGB Color Statistics (21 features)
For each color channel (R, G, B):
- Mean, Standard Deviation
- Min, Max, Median
- 25th percentile, 75th percentile

**Rationale**: Captures basic color distribution and lighting conditions.

### 2.2 HSV Color Space (6 features)
For each HSV channel:
- Mean, Standard Deviation

**Rationale**: HSV separates color information from intensity, making it more robust to lighting variations common in outdoor photography.

### 2.3 LAB Color Space (6 features)
For each LAB channel:
- Mean, Standard Deviation

**Rationale**: LAB is perceptually uniform and the A channel (green-red) is particularly useful for distinguishing vegetation.

### 2.4 NDVI - Normalized Difference Vegetation Index (5 features)

```python
NDVI = (Green - Red) / (Green + Red + ε)
```

Features extracted:
- Mean, Standard Deviation, Median
- 75th percentile, 25th percentile

**Rationale**: NDVI is a standard remote sensing metric. Healthy vegetation has high NDVI (>0.3) due to chlorophyll absorbing red light and reflecting green.

### 2.5 Green Ratio (3 features)

```python
Green_Ratio = Green / (Blue + Green + Red + ε)
```

Features: Mean, Std, 75th percentile

**Rationale**: Simple but effective metric for green vegetation presence.

### 2.6 Texture Features (7 features)
- Standard deviation of grayscale
- Mean absolute difference (vertical)
- Mean absolute difference (horizontal)

**Rationale**: Captures spatial patterns and texture complexity, distinguishing dense vs sparse vegetation.

### 2.7 Target Encoding (1 feature)
Label encode the target name (0-4) to help model distinguish between prediction tasks.

**Total: 48 + 1 = 49 features per sample**

## 3. Model Architecture

### 3.1 Base Models

**LightGBM (Primary)**
```python
LGBMRegressor(
    n_estimators=1000,
    max_depth=12,
    learning_rate=0.03,
    num_leaves=80,
    subsample=0.8,
    colsample_bytree=0.8
)
```

**XGBoost**
```python
XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8
)
```

**Random Forest**
```python
RandomForestRegressor(
    n_estimators=500,
    max_depth=25,
    min_samples_split=5,
    max_features='sqrt'
)
```

**Gradient Boosting**
```python
GradientBoostingRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8
)
```

**LightGBM (Variant)**
```python
LGBMRegressor(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=50,
    subsample=0.7,
    colsample_bytree=0.7
)
```

### 3.2 Cross-Validation Strategy

**GroupKFold with 5 splits**

Critical design choice: Group by `image_id` to prevent data leakage. Since each image has 5 targets, we must ensure the same image doesn't appear in both train and validation.

```python
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=image_ids)):
    # Train on train_idx, validate on val_idx
```

### 3.3 Ensemble Strategy

**Weighted Averaging**

Instead of simple averaging, we weight models by their inverse RMSE:

```python
weights = 1 / np.array([rmse_1, rmse_2, ..., rmse_5])
weights = weights / weights.sum()  # Normalize

final_prediction = np.average(all_predictions, axis=0, weights=weights)
```

This gives more influence to better-performing models.

### 3.4 Random Seed Variation

For each of 4 submissions, use different random seeds:
- Submission 1: MAGIC_SEED = 123
- Submission 2: MAGIC_SEED = 456
- Submission 3: MAGIC_SEED = 789
- Submission 4: MAGIC_SEED = 999

Seeds affect:
- Random Forest tree construction
- XGBoost/LightGBM row/column sampling
- Cross-validation fold assignments

This provides submission diversity and reduces overfitting to specific random initializations.

## 4. Training Pipeline

```
1. Load train.csv and test.csv
2. Extract features from all images
3. Encode target names
4. For each model:
   a. Initialize with (MAGIC_SEED + fold) for fold-specific seed
   b. 5-fold GroupKFold cross-validation
   c. Store out-of-fold predictions
   d. Average test predictions across folds
5. Compute CV RMSE for each model
6. Calculate ensemble weights (inverse RMSE)
7. Weighted average of all model predictions
8. Clip predictions to [0, ∞)
9. Create submission file
```

## 5. Design Decisions

### Why Not Deep Learning?

**Pros of Traditional ML:**
- Limited data (357 training images)
- Fast training (minutes vs hours)
- More interpretable
- Strong baseline with domain features

**When to use Deep Learning:**
- 10,000+ images
- Complex spatial patterns
- Transfer learning from ImageNet

### Why Ensemble?

- Reduces variance
- Combines different model biases
- XGBoost: handles non-linearity well
- LightGBM: fast, efficient with categorical
- Random Forest: robust to outliers
- Gradient Boosting: smooth predictions

### Why These Features?

- **NDVI**: Proven in agriculture/remote sensing
- **Color spaces**: Capture different aspects of vegetation
- **Texture**: Distinguishes dense vs sparse coverage
- **Statistics**: Robust to image position/scale

## 6. Validation Strategy

**Metric Alignment:** Our CV metric (RMSE) is related but not identical to competition metric (weighted R²). This is acceptable as:
- Both measure prediction accuracy
- RMSE is simpler for model selection
- Final leaderboard uses weighted R²

**Group-based splitting:** Critical to prevent overfitting. Without this, the model would see the same images in train and validation.

## 7. Potential Failure Modes

1. **Overfitting to specific image conditions** (lighting, camera angle)
2. **Covariate shift** if test images from different location/season
3. **Outliers** in biomass measurements (e.g., unusual weather events)
4. **Feature correlation** leading to redundant information

Mitigations:
- Group CV prevents image leakage
- Ensemble reduces overfitting
- Clipping predictions to [0, ∞) enforces physical constraints
- Multiple seeds for submission diversity
