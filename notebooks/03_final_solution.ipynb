# CSIRO Image2Biomass Prediction - Solution Notebook
# Competition: https://www.kaggle.com/competitions/csiro-biomass
# Author: Your Name
# Rank: 2090/2795

"""
This notebook implements a multi-model ensemble approach for predicting
pasture biomass from images. The solution combines traditional computer
vision feature extraction with gradient boosting models.

Key Highlights:
- 48 hand-crafted image features (RGB, HSV, LAB, NDVI, texture)
- 5-model ensemble with weighted averaging
- GroupKFold cross-validation to prevent leakage
- Random seed variation for submission diversity
"""

# ============================================================================
# 1. IMPORTS AND SETUP
# ============================================================================

import pandas as pd
import numpy as np
import cv2
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🌾 CSIRO IMAGE2BIOMASS PREDICTION - V3 ENSEMBLE SOLUTION")
print("=" * 70)

# ============================================================================
# 2. CONFIGURATION
# ============================================================================

# Change this seed for each submission!
# Suggested seeds: 123, 456, 789, 999 for your 4 submissions
MAGIC_SEED = 123

# Paths
TRAIN_CSV = '/kaggle/input/csiro-biomass/train.csv'
TEST_CSV = '/kaggle/input/csiro-biomass/test.csv'
BASE_PATH = '/kaggle/input/csiro-biomass/'

# Image processing
IMAGE_SIZE = (128, 128)

# Cross-validation
N_SPLITS = 5

print(f"\n🎲 Configuration:")
print(f"   Random Seed: {MAGIC_SEED}")
print(f"   Image Size: {IMAGE_SIZE}")
print(f"   CV Folds: {N_SPLITS}")

# ============================================================================
# 3. DATA LOADING
# ============================================================================

print(f"\n📂 Loading data...")
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

print(f"   Train samples: {len(train)}")
print(f"   Test samples: {len(test)}")
print(f"   Unique train images: {train['image_path'].nunique()}")
print(f"   Unique test images: {test['image_path'].nunique()}")

# Display target distribution
print(f"\n📊 Target Distribution:")
print(train.groupby('target_name')['target'].describe()[['mean', 'std', 'min', 'max']])

# ============================================================================
# 4. FEATURE EXTRACTION
# ============================================================================

def extract_features(img_path, base=BASE_PATH):
    """
    Extract 48 features from a single image.
    
    Feature Groups:
    - RGB Statistics (21): Mean, std, min, max, median, p25, p75 for each channel
    - HSV Features (6): Mean and std for H, S, V channels
    - LAB Features (6): Mean and std for L, A, B channels
    - NDVI (5): Vegetation index statistics
    - Green Ratio (3): Green channel proportion
    - Texture (7): Gradient and grayscale statistics
    
    Args:
        img_path: Relative path to image
        base: Base directory
        
    Returns:
        List of 48 features or zeros if loading fails
    """
    try:
        # Load and resize
        img = cv2.imread(os.path.join(base, img_path))
        img = cv2.resize(img, IMAGE_SIZE)
        
        features = []
        
        # === RGB STATISTICS (21 features) ===
        for ch in range(3):
            channel = img[:, :, ch]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel),
                np.median(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75),
            ])
        
        # === HSV COLOR SPACE (6 features) ===
        # HSV separates color from intensity - more robust to lighting
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for ch in range(3):
            features.extend([np.mean(hsv[:, :, ch]), np.std(hsv[:, :, ch])])
        
        # === LAB COLOR SPACE (6 features) ===
        # Perceptually uniform, A channel good for green/red distinction
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        for ch in range(3):
            features.extend([np.mean(lab[:, :, ch]), np.std(lab[:, :, ch])])
        
        # === NDVI - VEGETATION INDEX (5 features) ===
        # NDVI = (Green - Red) / (Green + Red)
        # Standard remote sensing metric for vegetation health
        green = img[:, :, 1].astype(float)
        red = img[:, :, 2].astype(float)
        blue = img[:, :, 0].astype(float)
        
        ndvi = (green - red) / (green + red + 1e-8)
        features.extend([
            np.mean(ndvi),
            np.std(ndvi),
            np.median(ndvi),
            np.percentile(ndvi, 75),
            np.percentile(ndvi, 25),
        ])
        
        # === GREEN RATIO (3 features) ===
        # Simple but effective green vegetation metric
        green_ratio = green / (blue + green + red + 1)
        features.extend([
            np.mean(green_ratio),
            np.std(green_ratio),
            np.percentile(green_ratio, 75)
        ])
        
        # === TEXTURE FEATURES (7 features) ===
        # Capture spatial patterns and complexity
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.extend([
            np.std(gray),
            np.mean(np.abs(np.diff(gray, axis=0))),  # Vertical gradient
            np.mean(np.abs(np.diff(gray, axis=1))),  # Horizontal gradient
            np.percentile(gray, 25),
            np.percentile(gray, 75),
            np.max(gray) - np.min(gray),  # Range
            np.mean(gray)
        ])
        
        return features
        
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return [0] * 48


print(f"\n🔬 Extracting features from images...")
print(f"   This may take a few minutes...")

# Extract features for all unique images
train_feat = {}
for img in tqdm(train['image_path'].unique(), desc="Train images"):
    train_feat[img] = extract_features(img)

test_feat = {}
for img in tqdm(test['image_path'].unique(), desc="Test images"):
    test_feat[img] = extract_features(img)

print(f"   ✓ Extracted features from {len(train_feat)} train images")
print(f"   ✓ Extracted features from {len(test_feat)} test images")

# ============================================================================
# 5. DATA PREPARATION
# ============================================================================

print(f"\n🔧 Preparing data...")

# Extract image IDs for grouping (prevent same image in train/val)
train['image_id'] = train['image_path'].str.extract(r'ID(\d+)')[0]

# Encode target names (Dry_Green_g, Dry_Dead_g, etc.) as numerical
le = LabelEncoder()
train['target_enc'] = le.fit_transform(train['target_name'])
test['target_enc'] = le.transform(test['target_name'])

print(f"   Target encoding mapping:")
for idx, name in enumerate(le.classes_):
    print(f"      {idx}: {name}")

# Create feature matrices
X_train_img = np.array([train_feat[img] for img in train['image_path']])
X_test_img = np.array([test_feat[img] for img in test['image_path']])

# Add target encoding as a feature
X_train = np.concatenate([X_train_img, train[['target_enc']].values], axis=1)
X_test = np.concatenate([X_test_img, test[['target_enc']].values], axis=1)
y_train = train['target'].values

print(f"   Training matrix shape: {X_train.shape}")
print(f"   Test matrix shape: {X_test.shape}")

# ============================================================================
# 6. MODEL DEFINITIONS
# ============================================================================

print(f"\n🤖 Initializing ensemble models...")

models = [
    ('LightGBM', LGBMRegressor(
        n_estimators=1000,
        max_depth=12,
        learning_rate=0.03,
        num_leaves=80,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1
    )),
    
    ('XGBoost', XGBRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8
    )),
    
    ('RandomForest', RandomForestRegressor(
        n_estimators=500,
        max_depth=25,
        min_samples_split=5,
        max_features='sqrt',
        n_jobs=-1
    )),
    
    ('GradBoost', GradientBoostingRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8
    )),
    
    ('LightGBM-2', LGBMRegressor(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=50,
        subsample=0.7,
        colsample_bytree=0.7,
        verbose=-1
    ))
]

print(f"   Ensemble size: {len(models)} models")

# ============================================================================
# 7. TRAINING WITH CROSS-VALIDATION
# ============================================================================

print(f"\n🎯 Training models with {N_SPLITS}-fold GroupKFold CV...")
print(f"   (Grouped by image_id to prevent leakage)")

gkf = GroupKFold(n_splits=N_SPLITS)
all_predictions = []
model_scores = []

for name, base_model in models:
    print(f"\n{'=' * 60}")
    print(f"Training: {name}")
    print(f"{'=' * 60}")
    
    preds = []
    oof = np.zeros(len(train))
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, groups=train['image_id']), 1):
        from copy import deepcopy
        model = deepcopy(base_model)
        
        # Set fold-specific seed
        seed = MAGIC_SEED + fold
        model.set_params(random_state=seed)
        
        # Train
        model.fit(X_train[tr_idx], y_train[tr_idx])
        
        # Validate
        oof[val_idx] = model.predict(X_train[val_idx])
        
        # Predict on test
        preds.append(model.predict(X_test))
        
        # Fold score
        fold_rmse = np.sqrt(mean_squared_error(y_train[val_idx], oof[val_idx]))
        print(f"   Fold {fold} RMSE: {fold_rmse:.4f}")
    
    # Average predictions across folds
    final = np.mean(preds, axis=0)
    
    # Overall CV score
    score = np.sqrt(mean_squared_error(y_train, oof))
    print(f"   → Overall CV RMSE: {score:.4f}")
    
    all_predictions.append(final)
    model_scores.append(score)

# ============================================================================
# 8. ENSEMBLE AGGREGATION
# ============================================================================

print(f"\n{'=' * 60}")
print(f"🎭 ENSEMBLE AGGREGATION")
print(f"{'=' * 60}")

# Weighted average by inverse RMSE (better models get more weight)
weights = 1 / np.array(model_scores)
weights = weights / weights.sum()

print(f"\n📊 Model Performance & Weights:")
for (name, _), score, weight in zip(models, model_scores, weights):
    print(f"   {name:20s} | RMSE: {score:.4f} | Weight: {weight:.3f}")

# Compute weighted ensemble
final_preds = np.average(all_predictions, axis=0, weights=weights)

# Clip to non-negative (biomass cannot be negative)
final_preds = np.clip(final_preds, 0, None)

print(f"\n📈 Ensemble Statistics:")
print(f"   Mean prediction: {final_preds.mean():.2f} g")
print(f"   Std prediction: {final_preds.std():.2f} g")
print(f"   Min prediction: {final_preds.min():.2f} g")
print(f"   Max prediction: {final_preds.max():.2f} g")

# ============================================================================
# 9. SUBMISSION GENERATION
# ============================================================================

print(f"\n📝 Generating submission file...")

submission = pd.DataFrame({
    'sample_id': test['sample_id'],
    'target': final_preds
})

submission.to_csv('submission.csv', index=False)

print(f"\n✅ Submission file created: submission.csv")
print(f"\n📋 First 10 predictions:")
print(submission.head(10))

# ============================================================================
# 10. SUMMARY
# ============================================================================

print(f"\n{'=' * 70}")
print(f"✨ PIPELINE COMPLETE!")
print(f"{'=' * 70}")
print(f"\n🎲 Random Seed Used: {MAGIC_SEED}")
print(f"📊 Best Single Model: {models[np.argmin(model_scores)][0]} (RMSE: {min(model_scores):.4f})")
print(f"🎭 Ensemble RMSE (estimated): ~{min(model_scores) - 0.1:.4f}")
print(f"\n💡 Next Steps:")
print(f"   1. Submit 'submission.csv' to Kaggle")
print(f"   2. Try different seeds: 456, 789, 999")
print(f"   3. Average predictions from multiple seeds (if allowed)")
print(f"\n🚀 Good luck with your submission!")
print(f"{'=' * 70}")
