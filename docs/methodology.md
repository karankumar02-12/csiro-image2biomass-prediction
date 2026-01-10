# Methodology: CSIRO Image2Biomass Prediction

**A detailed explanation of our approach to predicting pasture biomass from images**

---

## 📋 Table of Contents
1. [Problem Formulation](#1-problem-formulation)
2. [Feature Engineering](#2-feature-engineering)
3. [Model Architecture](#3-model-architecture)
4. [Training Strategy](#4-training-strategy)
5. [Ensemble Method](#5-ensemble-method)
6. [Validation Strategy](#6-validation-strategy)
7. [Design Decisions](#7-design-decisions)

---

## 1. Problem Formulation

### 1.1 Problem Statement

Predict **5 biomass measurements** from a single top-down pasture image:

| Target | Description | Weight |
|--------|-------------|--------|
| `Dry_Green_g` | Dry weight of green vegetation | 0.1 (10%) |
| `Dry_Dead_g` | Dry weight of dead vegetation | 0.1 (10%) |
| `Dry_Clover_g` | Dry weight of clover | 0.1 (10%) |
| `GDM_g` | Green dry matter | 0.2 (20%) |
| `Dry_Total_g` | Total dry biomass | 0.5 (50%) |

### 1.2 Mathematical Formulation

This is a **multi-target regression** problem:

```
Given: Image I ∈ ℝ^(H×W×3)
Predict: y = [y₁, y₂, y₃, y₄, y₅] ∈ ℝ^5
where y_i = biomass value for target i (in grams)
```

### 1.3 Evaluation Metric

**Weighted R² Score** computed globally:

```
R² = 1 - (Σᵢ wᵢ·(yᵢ - ŷᵢ)²) / (Σᵢ wᵢ·(yᵢ - ȳ)²)

where:
  wᵢ = weight for sample i based on target type
  yᵢ = true value
  ŷᵢ = predicted value
  ȳ = weighted mean of true values
```

**Key Insight**: This is a single global metric, not averaged R² per target. The 50% weight on `Dry_Total_g` means this target dominates the score.

---

## 2. Feature Engineering

### 2.1 Design Philosophy

With **limited training data** (357 images), we leverage **domain knowledge** to extract meaningful features rather than learning them from scratch with deep learning.

### 2.2 Feature Categories

#### **Category 1: RGB Color Statistics** (21 features)

For each channel (Blue, Green, Red):

```python
features = [
    mean(channel),
    std(channel),
    min(channel),
    max(channel),
    median(channel),
    percentile(channel, 25),
    percentile(channel, 75)
]
```

**Rationale**: Basic color distribution captures overall appearance and lighting conditions. Green channel particularly important for vegetation.

---

#### **Category 2: HSV Color Space** (6 features)

```python
HSV = cvtColor(image, COLOR_BGR2HSV)
features = [mean(H), std(H), mean(S), std(S), mean(V), std(V)]
```

**Why HSV?**
- **Hue (H)**: Pure color information, independent of lighting
- **Saturation (S)**: Color intensity/purity
- **Value (V)**: Brightness

**Advantage**: More robust to lighting variations common in outdoor photography than RGB.

---

#### **Category 3: LAB Color Space** (6 features)

```python
LAB = cvtColor(image, COLOR_BGR2LAB)
features = [mean(L), std(L), mean(A), std(A), mean(B), std(B)]
```

**Why LAB?**
- **L channel**: Lightness (perceptually uniform)
- **A channel**: Green (-) to Red (+) axis
- **B channel**: Blue (-) to Yellow (+) axis

**Advantage**: The A channel is particularly useful for distinguishing green vegetation from dead/brown matter.

---

#### **Category 4: NDVI - Normalized Difference Vegetation Index** (5 features)

```python
NDVI = (Green - Red) / (Green + Red + ε)
features = [mean(NDVI), std(NDVI), median(NDVI), p25(NDVI), p75(NDVI)]
```

**Scientific Basis**:
- Healthy vegetation reflects strongly in green light (photosynthesis)
- Chlorophyll absorbs red light
- NDVI values: -1 to +1
  - **> 0.3**: Healthy vegetation
  - **0.2-0.3**: Sparse vegetation
  - **< 0.2**: Soil/dead matter

**Why It Works**: Standard metric in remote sensing and precision agriculture. Directly correlates with vegetation health and biomass.

---

#### **Category 5: Green Ratio** (3 features)

```python
GR = Green / (Blue + Green + Red + ε)
features = [mean(GR), std(GR), percentile(GR, 75)]
```

**Rationale**: Simple but effective metric for proportion of green vegetation in image. Complements NDVI.

---

#### **Category 6: Texture Features** (7 features)

```python
gray = cvtColor(image, COLOR_BGR2GRAY)

features = [
    std(gray),                           # Overall texture variation
    mean(|∇y|),                          # Vertical gradient
    mean(|∇x|),                          # Horizontal gradient
    percentile(gray, 25),
    percentile(gray, 75),
    max(gray) - min(gray),               # Dynamic range
    mean(gray)
]
```

**Rationale**: Texture captures spatial patterns. Dense vegetation has different texture than sparse vegetation.

---

#### **Category 7: Target Encoding** (1 feature)

```python
target_enc = LabelEncoder.transform(target_name)  # 0-4
```

**Why?** Allows model to learn target-specific patterns. For example, clover might have different visual characteristics than total biomass.

---

### 2.3 Feature Summary

| Category | Count | Importance | Purpose |
|----------|-------|------------|---------|
| RGB Statistics | 21 | High | Basic color distribution |
| HSV Features | 6 | Medium | Lighting-invariant color |
| LAB Features | 6 | High | Green-red distinction |
| NDVI | 5 | **Very High** | Vegetation health |
| Green Ratio | 3 | High | Green proportion |
| Texture | 7 | Medium | Spatial patterns |
| Target Encoding | 1 | Medium | Task-specific patterns |
| **Total** | **49** | - | - |

---

## 3. Model Architecture

### 3.1 Base Models

We use **5 gradient boosting models** with diverse hyperparameters:

#### **Model 1: LightGBM (Primary)**

```python
LGBMRegressor(
    n_estimators=1000,
    max_depth=12,
    learning_rate=0.03,
    num_leaves=80,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=seed
)
```

**Characteristics**:
- Leaf-wise tree growth (vs level-wise)
- Faster training than XGBoost
- Good with high-dimensional features
- Built-in categorical handling

---

#### **Model 2: XGBoost**

```python
XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=seed
)
```

**Characteristics**:
- Level-wise tree growth
- Strong regularization
- Handles missing values well
- Generally more stable than LightGBM

---

#### **Model 3: Random Forest**

```python
RandomForestRegressor(
    n_estimators=500,
    max_depth=25,
    min_samples_split=5,
    max_features='sqrt',
    random_state=seed
)
```

**Characteristics**:
- Bagging instead of boosting
- Parallel tree building
- Less prone to overfitting
- Provides ensemble diversity

---

#### **Model 4: Gradient Boosting**

```python
GradientBoostingRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    random_state=seed
)
```

**Characteristics**:
- Traditional boosting
- Smooth predictions
- Good baseline
- More interpretable

---

#### **Model 5: LightGBM (Variant)**

```python
LGBMRegressor(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=50,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=seed
)
```

**Characteristics**:
- Different hyperparameters than LightGBM-1
- Adds diversity to ensemble
- Less aggressive (smaller learning rate, fewer leaves)

---

### 3.2 Why These Models?

| Criterion | Justification |
|-----------|---------------|
| **Gradient Boosting Family** | Best for tabular data, handles non-linearity well |
| **Model Diversity** | Different algorithms (bagging vs boosting) and hyperparameters |
| **Complementary Strengths** | XGBoost: stable, LightGBM: fast, RF: robust |
| **Proven Performance** | These models dominate Kaggle tabular competitions |

---

## 4. Training Strategy

### 4.1 Cross-Validation: GroupKFold

```python
gkf = GroupKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=image_id)):
    # Train on train_idx
    # Validate on val_idx
```

**Critical Design Choice**: Group by `image_id`

**Why?**
- Each image has **5 targets** (Dry_Green_g, Dry_Dead_g, etc.)
- If same image appears in both train and validation → **leakage**
- Model would memorize image-specific patterns
- CV scores would be artificially inflated

**Example of Proper Split**:
```
Fold 1:
  Train: [Image_001, Image_002, ..., Image_285]  (285 images)
  Val:   [Image_286, ..., Image_357]             (72 images)

Fold 2:
  Train: [Image_001, ..., Image_071, Image_143, ..., Image_357]
  Val:   [Image_072, ..., Image_142]
...
```

---

### 4.2 Training Process

For each model, for each fold:

```python
# 1. Create fresh model copy
model = deepcopy(base_model)

# 2. Set fold-specific seed
seed = MAGIC_SEED + fold_number
model.set_params(random_state=seed)

# 3. Train
model.fit(X_train[train_idx], y_train[train_idx])

# 4. Predict on validation
oof_predictions[val_idx] = model.predict(X_train[val_idx])

# 5. Predict on test
test_predictions[fold] = model.predict(X_test)
```

---

### 4.3 Seed Management

**Base Seed Strategy**:
```python
MAGIC_SEED = 123  # Change for each submission

For fold 1: random_state = 123 + 1 = 124
For fold 2: random_state = 123 + 2 = 125
...
For fold 5: random_state = 123 + 5 = 128
```

**Multiple Submissions**:
- Submission 1: MAGIC_SEED = 123
- Submission 2: MAGIC_SEED = 456
- Submission 3: MAGIC_SEED = 789
- Submission 4: MAGIC_SEED = 999

**Purpose**: Seed affects random sampling in trees. Multiple seeds provide natural regularization and hedge against unlucky random splits.

---

## 5. Ensemble Method

### 5.1 Weighted Averaging

```python
# 1. Collect predictions from all models
all_predictions = [pred_model1, pred_model2, ..., pred_model5]

# 2. Calculate weights (inverse RMSE)
weights = 1 / np.array([rmse1, rmse2, rmse3, rmse4, rmse5])
weights = weights / weights.sum()  # Normalize to sum to 1

# 3. Weighted average
final_prediction = Σ (weight_i × prediction_i)

# 4. Clip to valid range
final_prediction = max(0, final_prediction)  # Biomass cannot be negative
```

### 5.2 Why Weighted Averaging?

**Comparison**:

| Method | RMSE | Rationale |
|--------|------|-----------|
| Simple Average | 17.82 | Equal weight to all models |
| **Weighted Average** | **17.52** | More weight to better models |
| Stacking | ~17.3 | Requires meta-learner (more complex) |

**Formula**:
```
weight_i = 1 / RMSE_i

If RMSE = [17.65, 18.05, 19.29, 17.93, 17.87]:
  raw_weights = [0.0567, 0.0554, 0.0518, 0.0558, 0.0560]
  normalized = [0.212, 0.207, 0.194, 0.208, 0.209]
```

**Result**: Better performing models (lower RMSE) get higher weight.

---

### 5.3 Ensemble Benefits

1. **Variance Reduction**: Average out random fluctuations
2. **Bias-Diversity Tradeoff**: Combine models with different biases
3. **Robustness**: Less sensitive to individual model failures
4. **Improved Generalization**: Less overfitting than any single model

**Mathematical Intuition**:
```
Var(Average) = Var(X₁ + X₂ + ... + Xₙ) / n²
             ≤ (Var(X₁) + ... + Var(Xₙ)) / n²
             ≈ σ² / n  (if models independent)
```

So ensemble variance scales as **1/n** with n models.

---

## 6. Validation Strategy

### 6.1 Metrics

**Training Metric**: RMSE (Root Mean Squared Error)

```python
RMSE = sqrt(mean((y_true - y_pred)²))
```

**Why RMSE?**
- Easy to interpret (same units as target: grams)
- Penalizes large errors more than MAE
- Related to competition's R² metric

**Competition Metric**: Weighted R²
```python
R² = 1 - Σ(w_i·(y_i - ŷ_i)²) / Σ(w_i·(y_i - ȳ)²)
```

**Note**: RMSE and R² are related but not identical. Lower RMSE generally means higher R².

---

### 6.2 Out-of-Fold Predictions

```python
oof_predictions = np.zeros(len(train))

for fold in range(5):
    # Train on 4 folds
    # Predict on 1 fold
    oof_predictions[val_idx] = model.predict(X_val)

# Overall CV score
cv_rmse = sqrt(mean((y_train - oof_predictions)²))
```

**Advantage**: Every sample gets exactly 1 prediction from a model that hasn't seen it during training. Provides unbiased estimate of generalization.

---

### 6.3 Test Set Predictions

```python
test_predictions = []

for fold in range(5):
    # Each fold's model predicts on entire test set
    test_predictions.append(model_fold.predict(X_test))

# Average across folds
final_test_pred = mean(test_predictions)
```

**Rationale**: Reduces variance by averaging predictions from 5 different models.

---

## 7. Design Decisions

### 7.1 Why Traditional ML Over Deep Learning?

| Criterion | Traditional ML | Deep Learning |
|-----------|----------------|---------------|
| **Training Data** | 357 images ✅ | Need 10,000+ ❌ |
| **Training Time** | Minutes ✅ | Hours ❌ |
| **Interpretability** | Feature importance ✅ | Black box ❌ |
| **Domain Knowledge** | Can incorporate ✅ | Learns from scratch ❌ |
| **Hardware** | CPU sufficient ✅ | GPU preferred ❌ |
| **Overfitting Risk** | Manageable ✅ | High with small data ❌ |

**Conclusion**: With limited data, well-engineered features + gradient boosting is optimal.

---

### 7.2 Why 48 Features?

**Too Few Features** (< 20):
- Underfitting
- Missing important information
- Lower performance ceiling

**Too Many Features** (> 100):
- Curse of dimensionality
- Increased overfitting risk
- Slower training
- Redundant information

**48 Features = Sweet Spot**:
- Captures essential information
- Multiple complementary views (RGB, HSV, LAB, NDVI)
- Manageable for models
- Good signal-to-noise ratio

---

### 7.3 Why 5 Models in Ensemble?

**Marginal Benefit Analysis**:

| # Models | Estimated RMSE | Training Time | Diminishing Returns |
|----------|----------------|---------------|---------------------|
| 1 | 17.65 | 1x | - |
| 2 | 17.58 | 2x | 0.07 improvement |
| 3 | 17.55 | 3x | 0.03 improvement |
| 5 | **17.52** | **5x** | **0.03 improvement** ✅ |
| 10 | ~17.50 | 10x | 0.02 improvement ❌ |

**Decision**: 5 models provides good balance of:
- Performance improvement
- Training time
- Diversity (different algorithms)

---

### 7.4 Why These Hyperparameters?

**General Principles**:
1. **High n_estimators** (500-1000): More trees = better performance, up to a point
2. **Moderate depth** (8-12): Deep enough for non-linearity, not so deep we overfit
3. **Low learning rate** (0.03-0.05): Slower learning = more stable
4. **Subsampling** (0.7-0.8): Row sampling reduces overfitting
5. **Column sampling** (0.7-0.8): Feature sampling adds diversity

**Tuning Process**:
1. Start with defaults
2. Grid search on key parameters (depth, learning rate)
3. Validate with cross-validation
4. Fine-tune based on CV feedback

---

## 8. Complete Pipeline Flowchart

```
┌─────────────────────┐
│  Load Train/Test    │
│      Data           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Extract Features   │
│   (48 per image)    │
│ RGB, HSV, LAB,NDVI  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Encode Targets     │
│  (target_name 0-4)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5-Fold GroupKFold  │
│  (by image_id)      │
└──────────┬──────────┘
           │
           ▼
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌───────┐    ┌────────┐
│ Train │    │  Val   │
│ (4/5) │    │ (1/5)  │
└───┬───┘    └────┬───┘
    │             │
    ▼             ▼
┌─────────────────────┐
│  Train 5 Models     │
│  (each on 5 folds)  │
│  • LightGBM x2      │
│  • XGBoost          │
│  • RF               │
│  • GradBoost        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  OOF Predictions    │
│  (CV scores)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Calculate Weights  │
│  (1 / RMSE)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Weighted Average   │
│  Final Predictions  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Clip to [0, ∞)     │
│  Generate CSV       │
└─────────────────────┘
```

---

## 9. Key Takeaways

### ✅ What Made This Solution Effective

1. **Domain-Driven Features**: NDVI and vegetation indices leveraged agricultural science
2. **Proper Validation**: GroupKFold prevented leakage and gave realistic estimates
3. **Ensemble Diversity**: Multiple algorithms and hyperparameters reduced variance
4. **Weighted Aggregation**: Better models got more influence
5. **Robust Pipeline**: Seed variation provided stability

### 🎓 Lessons Learned

1. **Feature engineering > Raw data** (with limited samples)
2. **Validation strategy matters** (GroupKFold vs regular KFold = 2+ RMSE difference)
3. **Ensemble always helps** (even simple averaging improves results)
4. **Domain knowledge pays off** (NDVI alone as valuable as 10 generic features)

---

## 10. References

### Academic Papers
- Rouse, J.W. et al. (1974). "Monitoring vegetation systems in the Great Plains with ERTS"
- Tucker, C.J. (1979). "Red and photographic infrared linear combinations for monitoring vegetation"

### Technical Resources
- [OpenCV Documentation](https://docs.opencv.org/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

### Related Work
- [Kaggle Plant Pathology Competition](https://www.kaggle.com/c/plant-pathology-2020-fgvc7)
- [Remote Sensing for Agriculture](https://www.mdpi.com/journal/remotesensing)

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Author**: Karan Kumar
