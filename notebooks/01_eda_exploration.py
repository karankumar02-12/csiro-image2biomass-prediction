"""
CSIRO Image2Biomass - Exploratory Data Analysis
================================================

This script performs comprehensive exploratory data analysis on the CSIRO
biomass dataset to understand the data distribution, relationships, and
characteristics before building models.

Author: Karan Kumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("CSIRO IMAGE2BIOMASS - EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n1. LOADING DATA")
print("-" * 80)

TRAIN_CSV = '/kaggle/input/csiro-biomass/train.csv'
TEST_CSV = '/kaggle/input/csiro-biomass/test.csv'
BASE_PATH = '/kaggle/input/csiro-biomass/'

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")

print("\nFirst few rows of training data:")
print(train.head())

print("\nDataset Info:")
print(train.info())

# ============================================================================
# 2. TARGET ANALYSIS
# ============================================================================

print("\n\n2. TARGET VARIABLE ANALYSIS")
print("-" * 80)

# Target names
print("\nTarget Types:")
print(train['target_name'].value_counts())

# Statistical summary by target
print("\nStatistical Summary by Target:")
target_stats = train.groupby('target_name')['target'].describe()
print(target_stats)

# Distribution plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Biomass Targets', fontsize=16, fontweight='bold')

target_names = train['target_name'].unique()
for idx, target in enumerate(target_names):
    row = idx // 3
    col = idx % 3
    
    data = train[train['target_name'] == target]['target']
    
    axes[row, col].hist(data, bins=30, edgecolor='black', alpha=0.7)
    axes[row, col].set_title(f'{target}\nMean: {data.mean():.2f}g, Std: {data.std():.2f}g')
    axes[row, col].set_xlabel('Biomass (grams)')
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].grid(True, alpha=0.3)

# Remove empty subplot if odd number of targets
if len(target_names) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig('target_distributions.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: target_distributions.png")

# Box plots for comparison
plt.figure(figsize=(12, 6))
train.boxplot(column='target', by='target_name', figsize=(12, 6))
plt.suptitle('Biomass Distribution by Target Type', fontsize=14, fontweight='bold')
plt.xlabel('Target Type')
plt.ylabel('Biomass (grams)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('target_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: target_boxplots.png")

# ============================================================================
# 3. CORRELATION ANALYSIS
# ============================================================================

print("\n\n3. CORRELATION ANALYSIS")
print("-" * 80)

# Pivot to get correlations between targets
train_pivot = train.pivot_table(
    index='image_path',
    columns='target_name',
    values='target'
).reset_index()

print("\nCorrelation Matrix:")
correlation_matrix = train_pivot.drop('image_path', axis=1).corr()
print(correlation_matrix)

# Visualize correlation
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.3f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8}
)
plt.title('Correlation Between Biomass Targets', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('target_correlation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: target_correlation.png")

# Key insights
print("\nKey Correlations:")
for i in range(len(correlation_matrix)):
    for j in range(i+1, len(correlation_matrix)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.7:
            print(f"  {correlation_matrix.index[i]} ↔ {correlation_matrix.columns[j]}: {corr:.3f}")

# ============================================================================
# 4. IMAGE ANALYSIS
# ============================================================================

print("\n\n4. IMAGE ANALYSIS")
print("-" * 80)

# Image statistics
train['image_id'] = train['image_path'].str.extract(r'ID(\d+)')[0]
unique_images = train['image_id'].nunique()

print(f"\nTotal unique images: {unique_images}")
print(f"Samples per image: {len(train) / unique_images:.1f}")

# Load and analyze sample images
print("\nAnalyzing sample images...")

def analyze_image(img_path):
    """Analyze basic properties of an image."""
    try:
        full_path = os.path.join(BASE_PATH, img_path)
        img = cv2.imread(full_path)
        
        if img is None:
            return None
        
        return {
            'height': img.shape[0],
            'width': img.shape[1],
            'channels': img.shape[2],
            'mean_brightness': np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
            'std_brightness': np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        }
    except:
        return None

# Analyze first 20 images
sample_images = train['image_path'].unique()[:20]
image_stats = []

for img_path in sample_images:
    stats = analyze_image(img_path)
    if stats:
        image_stats.append(stats)

if image_stats:
    stats_df = pd.DataFrame(image_stats)
    print("\nImage Statistics:")
    print(stats_df.describe())
    
    # Plot brightness distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(stats_df['mean_brightness'], bins=15, edgecolor='black', alpha=0.7)
    plt.xlabel('Mean Brightness')
    plt.ylabel('Frequency')
    plt.title('Distribution of Image Brightness')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(stats_df['mean_brightness'], stats_df['std_brightness'], alpha=0.6)
    plt.xlabel('Mean Brightness')
    plt.ylabel('Std Brightness')
    plt.title('Brightness Mean vs Std')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('image_brightness_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: image_brightness_analysis.png")

# Display sample images with their biomass values
print("\nCreating sample image visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Sample Images with Biomass Values', fontsize=16, fontweight='bold')

sample_img_paths = train['image_path'].unique()[:6]

for idx, img_path in enumerate(sample_img_paths):
    row = idx // 3
    col = idx % 3
    
    try:
        img = cv2.imread(os.path.join(BASE_PATH, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get biomass values for this image
        img_data = train[train['image_path'] == img_path]
        biomass_text = '\n'.join([
            f"{row['target_name']}: {row['target']:.1f}g"
            for _, row in img_data.iterrows()
        ])
        
        axes[row, col].imshow(img_rgb)
        axes[row, col].set_title(f"Image {idx+1}\n{biomass_text}", fontsize=8)
        axes[row, col].axis('off')
    except:
        axes[row, col].text(0.5, 0.5, 'Image not found', 
                           ha='center', va='center')
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sample_images.png")

# ============================================================================
# 5. MISSING VALUES & DATA QUALITY
# ============================================================================

print("\n\n5. DATA QUALITY CHECK")
print("-" * 80)

print("\nMissing values in training set:")
missing = train.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values found")
else:
    print(missing[missing > 0])

print("\nMissing values in test set:")
missing_test = test.isnull().sum()
if missing_test.sum() == 0:
    print("  ✓ No missing values found")
else:
    print(missing_test[missing_test > 0])

# Check for outliers
print("\nOutlier Detection (values > 3 std from mean):")
for target_name in train['target_name'].unique():
    data = train[train['target_name'] == target_name]['target']
    mean = data.mean()
    std = data.std()
    outliers = data[(data < mean - 3*std) | (data > mean + 3*std)]
    
    if len(outliers) > 0:
        print(f"  {target_name}: {len(outliers)} outliers")
        print(f"    Range: {outliers.min():.2f} - {outliers.max():.2f}g")
    else:
        print(f"  {target_name}: No outliers")

# ============================================================================
# 6. TRAIN-TEST COMPARISON
# ============================================================================

print("\n\n6. TRAIN-TEST COMPARISON")
print("-" * 80)

print("\nTarget distribution comparison:")
print("\nTrain set:")
print(train['target_name'].value_counts())
print("\nTest set:")
print(test['target_name'].value_counts())

# Check if target distributions match
train_dist = train['target_name'].value_counts(normalize=True).sort_index()
test_dist = test['target_name'].value_counts(normalize=True).sort_index()

print("\nProportion comparison:")
comparison = pd.DataFrame({
    'Train': train_dist,
    'Test': test_dist,
    'Difference': (train_dist - test_dist).abs()
})
print(comparison)

# ============================================================================
# 7. SUMMARY STATISTICS
# ============================================================================

print("\n\n7. SUMMARY STATISTICS")
print("-" * 80)

summary = pd.DataFrame({
    'Total Samples': [len(train)],
    'Unique Images': [train['image_id'].nunique()],
    'Target Types': [train['target_name'].nunique()],
    'Mean Biomass': [train['target'].mean()],
    'Std Biomass': [train['target'].std()],
    'Min Biomass': [train['target'].min()],
    'Max Biomass': [train['target'].max()]
})

print("\nDataset Summary:")
print(summary.T)

# Competition metric weights
print("\n\nCompetition Metric Weights:")
weights = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g': 0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g': 0.2,
    'Dry_Total_g': 0.5
}

for target, weight in weights.items():
    print(f"  {target:15s}: {weight:.1f} (or {weight*100:.0f}%)")

print("\n" + "=" * 80)
print("KEY INSIGHTS:")
print("=" * 80)

print("""
1. TARGETS:
   - 5 different biomass measurements per image
   - Dry_Total_g has highest weight (50%) in competition metric
   - Strong correlations expected between related targets

2. IMAGES:
   - Each image has 5 associated measurements
   - GroupKFold CV essential to prevent leakage
   - Variable lighting conditions visible

3. DATA QUALITY:
   - No missing values
   - Some outliers present in certain targets
   - Train-test distributions appear balanced

4. MODELING CONSIDERATIONS:
   - Multi-target prediction problem
   - Image features crucial (color, texture, vegetation indices)
   - Consider target relationships in modeling
   - Weight predictions by competition metric

NEXT STEPS:
→ Feature engineering (RGB, HSV, LAB, NDVI)
→ Build baseline models
→ Implement ensemble strategy
""")

print("\n✅ EDA Complete! Check the generated PNG files for visualizations.")
print("=" * 80)
