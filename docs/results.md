# Results & Performance Analysis

## 📊 Competition Performance

### Final Standing
- **Rank**: 2090 / 2795
- **Percentile**: Top 75%
- **Competition**: CSIRO - Image2Biomass Prediction
- **Evaluation Metric**: Weighted R² Score

### Timeline
- **Start Date**: October 28, 2025
- **Final Submission**: January 28, 2026
- **Total Duration**: 3 months

---

## 🎯 Model Performance

### Cross-Validation Results

All models evaluated using **5-Fold GroupKFold** cross-validation (grouped by image_id to prevent leakage).

#### Individual Model Performance

| Model | CV RMSE | Rank | Weight in Ensemble |
|-------|---------|------|-------------------|
| **XGBoost** | 17.6503 | 🥇 1st | 0.212 |
| **LightGBM-2** | 17.8744 | 🥈 2nd | 0.209 |
| **Gradient Boosting** | 17.9334 | 🥉 3rd | 0.208 |
| **LightGBM** | 18.0528 | 4th | 0.207 |
| **Random Forest** | 19.2901 | 5th | 0.194 |

#### Ensemble Performance

| Metric | Value |
|--------|-------|
| **Ensemble RMSE** | ~17.5 |
| **Improvement over best single model** | 0.15 RMSE |
| **Improvement over Random Forest** | 1.79 RMSE |

---

## 🔬 Detailed Analysis

### 1. Model Selection Rationale

#### Why These Models?

**XGBoost** (Best Performer - 17.65 RMSE)
- Excellent handling of non-linear relationships
- Robust regularization prevents overfitting
- Efficient with tabular features
- **Key strength**: Balanced performance across all targets

**LightGBM** (Close Second - 17.87 / 18.05 RMSE)
- Fastest training time
- Leaf-wise tree growth
- Good with high-dimensional features
- **Key strength**: Efficient memory usage, strong on texture features

**Gradient Boosting** (Third - 17.93 RMSE)
- Traditional but reliable
- Smooth predictions
- Less prone to overfitting than Random Forest
- **Key strength**: Stable predictions, good baseline

**Random Forest** (Fifth - 19.29 RMSE)
- Provides diversity in ensemble
- Different learning paradigm (bagging vs boosting)
- Less sensitive to hyperparameters
- **Key strength**: Adds robustness to ensemble

### 2. Cross-Validation Strategy

#### GroupKFold - Critical Design Choice

```
Fold 1: [Image_001, Image_002, ...] → Train | [Image_050, ...] → Validate
Fold 2: [Image_001, Image_050, ...] → Train | [Image_100, ...] → Validate
...
```

**Why GroupKFold?**
- Each image has 5 targets (Dry_Green_g, Dry_Dead_g, etc.)
- Same image must NOT appear in both train and validation
- Prevents information leakage
- More realistic estimate of generalization

**Impact**: Without GroupKFold, CV scores would be artificially inflated by ~2-3 RMSE points.

### 3. Feature Performance Analysis

#### Feature Importance (Top 15)

Based on Random Forest feature importance analysis:

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | NDVI_mean | 0.0842 | Vegetation Index |
| 2 | Green_mean | 0.0756 | RGB |
| 3 | GreenRatio_mean | 0.0689 | Vegetation Index |
| 4 | LAB_A_mean | 0.0621 | LAB Color |
| 5 | NDVI_p75 | 0.0587 | Vegetation Index |
| 6 | Saturation_mean | 0.0543 | HSV |
| 7 | Green_std | 0.0498 | RGB |
| 8 | Red_mean | 0.0476 | RGB |
| 9 | LAB_A_std | 0.0445 | LAB Color |
| 10 | ExG_mean | 0.0412 | Vegetation Index |
| 11 | Value_mean | 0.0389 | HSV |
| 12 | Green_median | 0.0367 | RGB |
| 13 | Gray_std | 0.0334 | Texture |
| 14 | NDVI_std | 0.0321 | Vegetation Index |
| 15 | Green_p75 | 0.0298 | RGB |

**Key Insights:**
- **Vegetation indices dominate** top positions (NDVI, Green Ratio, ExG)
- **Green channel** most important RGB channel
- **LAB A channel** (green-red axis) highly predictive
- **HSV Saturation** captures vegetation intensity

#### Feature Category Performance

| Category | Avg Importance | Description |
|----------|----------------|-------------|
| **Vegetation Indices** | 0.055 | NDVI, Green Ratio, ExG |
| **RGB Statistics** | 0.042 | Mean, std, percentiles |
| **LAB Color** | 0.038 | Perceptually uniform space |
| **HSV Color** | 0.035 | Hue, Saturation, Value |
| **Texture** | 0.028 | Gradients, edge features |
| **Color Histogram** | 0.022 | Distribution features |

---

## 📈 Performance by Target Type

### Target-Specific Analysis

Competition weights: Dry_Total_g (50%), GDM_g (20%), Others (10% each)

| Target | CV RMSE | Relative Error | Notes |
|--------|---------|----------------|-------|
| **Dry_Total_g** | 22.45 | ±15.2% | Highest weight, good performance |
| **GDM_g** | 18.73 | ±18.5% | Second highest weight |
| **Dry_Green_g** | 12.34 | ±21.3% | Lower absolute values |
| **Dry_Dead_g** | 14.89 | ±19.8% | More variable |
| **Dry_Clover_g** | 8.92 | ±24.6% | Often near zero |

**Observations:**
- Lower RMSE for targets with smaller absolute values
- Relative error higher for sparse targets (Clover)
- Dry_Total_g performance critical (50% of score)

---

## 🎲 Seed Variation Experiments

### Random Seed Impact

Tested 4 different random seeds for submission diversity:

| Seed | Ensemble RMSE | XGBoost RMSE | LightGBM RMSE | Notes |
|------|---------------|--------------|---------------|-------|
| **123** | 17.52 | 17.65 | 18.05 | Baseline |
| **456** | 17.48 | 17.61 | 17.99 | Best overall |
| **789** | 17.55 | 17.68 | 18.08 | Similar to baseline |
| **999** | 17.51 | 17.63 | 18.02 | Stable |

**Variance Analysis:**
- Standard deviation across seeds: **0.031 RMSE**
- Low variance indicates stable solution
- Seed 456 performed slightly better

**Recommendation**: Submit multiple seeds to hedge against unlucky random splits.

---

## 🔍 Error Analysis

### Prediction Distribution

```
Statistics on Final Predictions:
- Mean: 23.45 g
- Median: 18.32 g
- Std Dev: 15.67 g
- Min: 0.00 g (clipped)
- Max: 89.23 g
- Skewness: 1.23 (right-skewed)
```

### Common Prediction Errors

#### Overestimation Cases
- **Dense green pastures with shadows**: Model confused by lighting variations
- **Solution**: More robust color normalization

#### Underestimation Cases
- **Sparse but tall vegetation**: Height not captured in top-down images
- **Solution**: 3D features or multi-angle images

#### High Variance Predictions
- **Images with mixed vegetation types**: Clover vs grass distinction difficult
- **Solution**: Segmentation-based approach

---

## 💡 What Worked Well

### ✅ Successful Strategies

1. **Hand-Crafted Features > Raw Pixels**
   - NDVI and vegetation indices crucial
   - Domain knowledge paid off
   - Faster training than deep learning

2. **Ensemble Diversity**
   - Combining 5 different models reduced variance
   - Weighted averaging by inverse RMSE optimal
   - Each model contributed unique insights

3. **Proper Cross-Validation**
   - GroupKFold prevented overfitting
   - CV scores correlated well with leaderboard
   - Realistic performance estimates

4. **Feature Engineering**
   - Multiple color spaces (RGB, HSV, LAB)
   - Vegetation-specific indices
   - Texture and gradient features

5. **Seed Variation**
   - Multiple submissions with different seeds
   - Reduced impact of unlucky random splits

---

## ❌ What Didn't Work

### Approaches That Failed

1. **Raw Pixel Features**
   - Too high dimensional (128×128×3 = 49,152)
   - Overfitting despite regularization
   - Slower training with worse performance

2. **Simple Averaging**
   - Equal-weight ensemble: 17.82 RMSE
   - Weighted ensemble: 17.52 RMSE
   - **Improvement**: 0.30 RMSE by weighting

3. **Single Model Approach**
   - Best single model: 17.65 RMSE
   - Ensemble: 17.52 RMSE
   - **Improvement**: 0.13 RMSE

4. **Ignoring Group Structure**
   - Regular KFold: 15.23 RMSE (optimistically biased)
   - GroupKFold: 17.52 RMSE (realistic)
   - **Reality check**: +2.29 RMSE

---

## 🚀 Potential Improvements

### Short-Term Wins (Expected +5-10% improvement)

1. **Feature Selection**
   - Remove highly correlated features (|r| > 0.9)
   - Keep top 30-40 features by importance
   - Reduce noise and overfitting

2. **Hyperparameter Tuning**
   - Bayesian optimization for each model
   - Currently using manual tuning
   - Expected gain: 0.2-0.5 RMSE

3. **Stacking Ensemble**
   - Use meta-learner on top of base models
   - Linear regression or Ridge as meta-model
   - Learns optimal combination weights

4. **Target-Specific Models**
   - Train separate models for each target
   - Exploit target-specific patterns
   - More flexible than single multi-task model

### Medium-Term Enhancements (Expected +10-20% improvement)

1. **Image Segmentation**
   - Separate green vs dead vegetation
   - Target-specific spatial features
   - K-means or watershed segmentation

2. **Multi-Scale Features**
   - Extract features at multiple resolutions
   - Capture both fine and coarse patterns
   - Pyramid approach: 64×64, 128×128, 256×256

3. **Temporal Features** (if available)
   - Growth stage information
   - Season indicators
   - Weather data integration

4. **Semi-Supervised Learning**
   - Leverage unlabeled pasture images
   - Self-training or pseudo-labeling
   - Increases effective training data

### Long-Term Innovations (Expected +20-40% improvement)

1. **Transfer Learning (CNNs)**
   - Fine-tune ResNet50 or EfficientNet
   - Pre-trained on ImageNet
   - Learn spatial patterns automatically
   - **Requirement**: More training data (1000+ images)

2. **Multi-Task Learning**
   - Shared representations across targets
   - Joint prediction of all 5 targets
   - Exploit target correlations

3. **Attention Mechanisms**
   - Learn which image regions matter
   - Focus on vegetation areas
   - Ignore irrelevant background

4. **3D Vision**
   - Stereo images for height estimation
   - Point cloud features
   - Better capture biomass volume

5. **Ensemble of Deep + Traditional**
   - Combine CNN features with hand-crafted
   - Best of both worlds
   - Deep learning for spatial, traditional for domain knowledge

---

## 📊 Comparison with Baselines

### Baseline Models

| Model | RMSE | Description |
|-------|------|-------------|
| Mean Prediction | 45.23 | Predict mean of training set |
| Linear Regression (RGB mean only) | 38.67 | 3 features |
| Random Forest (RGB stats) | 28.45 | 21 features |
| **Our Solution** | **17.52** | 48 features + ensemble |
| Top Leaderboard (estimated) | ~12-14 | Likely deep learning |

**Performance Gains:**
- vs Mean: -61.3% error
- vs Linear: -54.7% error
- vs Basic RF: -38.4% error

---

## 🎓 Key Learnings

### Technical Insights

1. **Domain knowledge is powerful**
   - NDVI alone worth 10+ generic features
   - Understanding vegetation optics helped

2. **Proper validation is critical**
   - GroupKFold revealed true performance
   - Saved from overfitting trap

3. **Ensemble > Single Model**
   - Even simple averaging helps
   - Weighted averaging helps more
   - Diversity is key

4. **Feature engineering matters**
   - With limited data, hand-crafted beats deep learning
   - Multiple color spaces capture different aspects
   - Texture complements color

### Process Learnings

1. **Start simple, iterate**
   - Baseline → Feature engineering → Ensemble
   - Each step validated with CV

2. **Visualize everything**
   - EDA revealed target correlations
   - Feature importance guided selection
   - Error analysis showed weaknesses

3. **Document experiments**
   - Tracked what worked and what didn't
   - Seed experiments quantified variance
   - Reproducible results

---

## 📝 Competition Insights

### What Top Teams Likely Did

Based on competition patterns and best practices:

1. **Deep Learning Approaches**
   - CNN architectures (ResNet, EfficientNet)
   - Transfer learning from ImageNet
   - Data augmentation (rotation, flip, color jitter)

2. **Larger Ensembles**
   - 10-20 models instead of 5
   - Multiple architectures (CNN + traditional)
   - Pseudo-labeling on test set

3. **Advanced Features**
   - Superpixel segmentation
   - Texture operators (LBP, Gabor filters)
   - Spatial pyramid pooling

4. **External Data**
   - Similar pasture datasets
   - Pre-training on agricultural images
   - Weather/soil data if available

### Our Competitive Advantage

- **Fast iteration**: Traditional ML trains in minutes
- **Interpretable**: Know which features matter
- **Robust**: Lower variance than deep learning with small data
- **Efficient**: No GPU required

---

## 🎯 Conclusion

### Summary

Our solution achieved **Top 75%** ranking using:
- ✅ Smart feature engineering (48 features)
- ✅ Proper cross-validation (GroupKFold)
- ✅ Diverse ensemble (5 models)
- ✅ Weighted aggregation
- ✅ Seed variation for robustness

### Final Thoughts

This was a **solid traditional ML solution** that:
- Demonstrated strong fundamentals
- Balanced speed vs performance
- Provided interpretable results
- Served as excellent baseline

For production deployment or further improvement, the next step would be **transfer learning with CNNs** while maintaining the robust validation strategy developed here.

---

## 📚 References

### Papers & Resources
- [NDVI in Agriculture](https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index)
- [Color Spaces for Image Processing](https://opencv.org/)
- [Ensemble Learning Methods](https://scikit-learn.org/stable/modules/ensemble.html)

### Code & Data
- [Competition Page](https://www.kaggle.com/competitions/csiro-biomass)
- [GitHub Repository](https://github.com/karankumar02-12/csiro-image2biomass-prediction)

---

**Last Updated**: January 2026  
**Author**: Karan Kumar
