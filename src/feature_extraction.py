# ===== src/feature_extraction.py =====
"""Image feature extraction functions for biomass prediction."""

import cv2
import numpy as np
import os
from typing import List, Optional


def extract_rgb_features(img: np.ndarray) -> List[float]:
    """
    Extract statistical features from RGB channels.
    
    Args:
        img: Input image in BGR format (OpenCV default)
        
    Returns:
        List of 21 features (7 per channel: mean, std, min, max, median, p25, p75)
    """
    features = []
    
    for ch in range(3):  # B, G, R channels
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
    
    return features


def extract_hsv_features(img: np.ndarray) -> List[float]:
    """
    Extract features from HSV color space.
    
    Args:
        img: Input image in BGR format
        
    Returns:
        List of 6 features (mean and std for H, S, V)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = []
    
    for ch in range(3):
        features.extend([
            np.mean(hsv[:, :, ch]),
            np.std(hsv[:, :, ch])
        ])
    
    return features


def extract_lab_features(img: np.ndarray) -> List[float]:
    """
    Extract features from LAB color space.
    
    Args:
        img: Input image in BGR format
        
    Returns:
        List of 6 features (mean and std for L, A, B)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    features = []
    
    for ch in range(3):
        features.extend([
            np.mean(lab[:, :, ch]),
            np.std(lab[:, :, ch])
        ])
    
    return features


def extract_ndvi_features(img: np.ndarray) -> List[float]:
    """
    Extract NDVI (Normalized Difference Vegetation Index) features.
    
    NDVI = (Green - Red) / (Green + Red)
    
    Args:
        img: Input image in BGR format
        
    Returns:
        List of 5 NDVI statistics
    """
    green = img[:, :, 1].astype(float)
    red = img[:, :, 2].astype(float)
    
    # Calculate NDVI with epsilon to avoid division by zero
    ndvi = (green - red) / (green + red + 1e-8)
    
    features = [
        np.mean(ndvi),
        np.std(ndvi),
        np.median(ndvi),
        np.percentile(ndvi, 75),
        np.percentile(ndvi, 25),
    ]
    
    return features


def extract_green_ratio_features(img: np.ndarray) -> List[float]:
    """
    Extract green ratio features.
    
    Green Ratio = Green / (Blue + Green + Red)
    
    Args:
        img: Input image in BGR format
        
    Returns:
        List of 3 green ratio statistics
    """
    blue = img[:, :, 0].astype(float)
    green = img[:, :, 1].astype(float)
    red = img[:, :, 2].astype(float)
    
    green_ratio = green / (blue + green + red + 1)
    
    features = [
        np.mean(green_ratio),
        np.std(green_ratio),
        np.percentile(green_ratio, 75)
    ]
    
    return features


def extract_texture_features(img: np.ndarray) -> List[float]:
    """
    Extract texture and gradient features.
    
    Args:
        img: Input image in BGR format
        
    Returns:
        List of 7 texture features
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = [
        np.std(gray),
        np.mean(np.abs(np.diff(gray, axis=0))),  # Vertical gradient
        np.mean(np.abs(np.diff(gray, axis=1))),  # Horizontal gradient
    ]
    
    return features


def extract_image_features(
    img_path: str,
    base_path: str = '/kaggle/input/csiro-biomass/',
    image_size: tuple = (128, 128)
) -> List[float]:
    """
    Extract all features from a single image.
    
    Args:
        img_path: Relative path to the image
        base_path: Base directory containing images
        image_size: Target size for resizing (width, height)
        
    Returns:
        List of 48 features, or zeros if image loading fails
    """
    try:
        # Load and resize image
        full_path = os.path.join(base_path, img_path)
        img = cv2.imread(full_path)
        
        if img is None:
            raise ValueError(f"Failed to load image: {full_path}")
        
        img = cv2.resize(img, image_size)
        
        # Extract all feature groups
        features = []
        features.extend(extract_rgb_features(img))        # 21 features
        features.extend(extract_hsv_features(img))        # 6 features
        features.extend(extract_lab_features(img))        # 6 features
        features.extend(extract_ndvi_features(img))       # 5 features
        features.extend(extract_green_ratio_features(img))  # 3 features
        features.extend(extract_texture_features(img))    # 3 features (total would be 44, adding 4 more)
        
        # Add 4 more texture features to reach 48
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.extend([
            np.percentile(gray, 25),
            np.percentile(gray, 75),
            np.max(gray) - np.min(gray),  # Range
            np.mean(gray)
        ])
        
        return features
        
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return [0] * 48
