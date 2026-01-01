# ===== src/utils.py =====
"""Utility functions for data processing and submission generation."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple
from tqdm import tqdm


def load_and_prepare_data(
    train_csv: str,
    test_csv: str
) -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """
    Load and prepare train/test data.
    
    Args:
        train_csv: Path to training CSV
        test_csv: Path to test CSV
        
    Returns:
        Tuple of (train_df, test_df, label_encoder)
    """
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    
    # Extract image IDs for grouping
    train['image_id'] = train['image_path'].str.extract(r'ID(\d+)')[0]
    
    # Encode target names
    le = LabelEncoder()
    train['target_enc'] = le.fit_transform(train['target_name'])
    test['target_enc'] = le.transform(test['target_name'])
    
    return train, test, le


def extract_features_from_images(
    image_paths: pd.Series,
    feature_extractor,
    base_path: str,
    desc: str = "Extracting features"
) -> Dict[str, list]:
    """
    Extract features from multiple images.
    
    Args:
        image_paths: Series of image paths
        feature_extractor: Function to extract features from single image
        base_path: Base directory for images
        desc: Description for progress bar
        
    Returns:
        Dictionary mapping image_path to feature list
    """
    features = {}
    unique_paths = image_paths.unique()
    
    for img_path in tqdm(unique_paths, desc=desc):
        features[img_path] = feature_extractor(img_path, base_path)
    
    return features


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_dict: Dict[str, list]
) -> np.ndarray:
    """
    Create feature matrix from extracted features and encoded targets.
    
    Args:
        df: DataFrame with image_path and target_enc columns
        feature_dict: Dictionary of image_path -> features
        
    Returns:
        Feature matrix as numpy array
    """
    # Get image features
    X_img = np.array([feature_dict[img] for img in df['image_path']])
    
    # Concatenate with target encoding
    X = np.concatenate([X_img, df[['target_enc']].values], axis=1)
    
    return X


def generate_submission(
    predictions: np.ndarray,
    test_df: pd.DataFrame,
    output_path: str = 'submission.csv'
) -> pd.DataFrame:
    """
    Generate submission file.
    
    Args:
        predictions: Model predictions
        test_df: Test dataframe with sample_id
        output_path: Path to save submission CSV
        
    Returns:
        Submission dataframe
    """
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'target': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"\n✅ Submission saved to {output_path}")
    
    return submission


def print_submission_summary(submission: pd.DataFrame, magic_seed: int):
    """Print submission summary."""
    print("\n📋 Predictions Summary:")
    print(submission.head())
    print(f"\n🎲 Seed: {magic_seed}")
    print(f"📊 Prediction Stats:")
    print(f"   Mean: {submission['target'].mean():.2f}")
    print(f"   Std: {submission['target'].std():.2f}")
    print(f"   Min: {submission['target'].min():.2f}")
    print(f"   Max: {submission['target'].max():.2f}")
