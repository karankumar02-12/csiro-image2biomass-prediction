# ===== src/config.py =====
"""Configuration and hyperparameters for the CSIRO Biomass prediction pipeline."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Config:
    """Global configuration for the pipeline."""
    
    # Data paths
    TRAIN_CSV: str = '/kaggle/input/csiro-biomass/train.csv'
    TEST_CSV: str = '/kaggle/input/csiro-biomass/test.csv'
    BASE_IMAGE_PATH: str = '/kaggle/input/csiro-biomass/'
    
    # Image processing
    IMAGE_SIZE: tuple = (128, 128)
    
    # Cross-validation
    N_SPLITS: int = 5
    
    # Random seed
    MAGIC_SEED: int = 123
    
    # Model hyperparameters
    LIGHTGBM_PARAMS: Dict[str, Any] = None
    XGBOOST_PARAMS: Dict[str, Any] = None
    RANDOM_FOREST_PARAMS: Dict[str, Any] = None
    GRADIENT_BOOST_PARAMS: Dict[str, Any] = None
    LIGHTGBM2_PARAMS: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize model hyperparameters."""
        self.LIGHTGBM_PARAMS = {
            'n_estimators': 1000,
            'max_depth': 12,
            'learning_rate': 0.03,
            'num_leaves': 80,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': -1
        }
        
        self.XGBOOST_PARAMS = {
            'n_estimators': 1000,
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        self.RANDOM_FOREST_PARAMS = {
            'n_estimators': 500,
            'max_depth': 25,
            'min_samples_split': 5,
            'max_features': 'sqrt',
            'n_jobs': -1
        }
        
        self.GRADIENT_BOOST_PARAMS = {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8
        }
        
        self.LIGHTGBM2_PARAMS = {
            'n_estimators': 800,
            'max_depth': 8,
            'learning_rate': 0.05,
            'num_leaves': 50,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'verbose': -1
        }
