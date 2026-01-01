# ===== src/models.py =====
"""Model training and ensemble creation."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from typing import List, Tuple, Dict
from copy import deepcopy


def train_single_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    magic_seed: int = 123
) -> Tuple[List[np.ndarray], np.ndarray, float]:
    """
    Train a single model with cross-validation.
    
    Args:
        model: Sklearn-compatible model instance
        X_train: Training features
        y_train: Training targets
        groups: Group labels for GroupKFold
        n_splits: Number of CV folds
        magic_seed: Random seed base
        
    Returns:
        Tuple of (test_predictions_per_fold, out_of_fold_predictions, cv_rmse)
    """
    gkf = GroupKFold(n_splits=n_splits)
    test_preds = []
    oof_preds = np.zeros(len(y_train))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, groups=groups), 1):
        # Create a fresh copy of the model for each fold
        fold_model = deepcopy(model)
        
        # Set random state with fold-specific seed
        seed = magic_seed + fold
        fold_model.set_params(random_state=seed)
        
        # Train
        fold_model.fit(X_train[train_idx], y_train[train_idx])
        
        # Predict on validation set
        oof_preds[val_idx] = fold_model.predict(X_train[val_idx])
        
        # Store fold predictions (for test set later)
        test_preds.append(fold_model)
    
    # Calculate CV score
    cv_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    
    return test_preds, oof_preds, cv_rmse


def create_ensemble_models(config) -> List[Tuple[str, object]]:
    """
    Create list of models for ensemble.
    
    Args:
        config: Config object with hyperparameters
        
    Returns:
        List of (model_name, model_instance) tuples
    """
    models = [
        ('LightGBM', LGBMRegressor(**config.LIGHTGBM_PARAMS)),
        ('XGBoost', XGBRegressor(**config.XGBOOST_PARAMS)),
        ('RandomForest', RandomForestRegressor(**config.RANDOM_FOREST_PARAMS)),
        ('GradBoost', GradientBoostingRegressor(**config.GRADIENT_BOOST_PARAMS)),
        ('LightGBM-2', LGBMRegressor(**config.LIGHTGBM2_PARAMS))
    ]
    
    return models


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    groups: np.ndarray,
    config
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Train ensemble of models and generate predictions.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        groups: Group labels for CV
        config: Configuration object
        
    Returns:
        Tuple of (final_predictions, model_scores_dict)
    """
    models = create_ensemble_models(config)
    all_predictions = []
    model_scores = {}
    
    print(f"\n🎲 Using MAGIC_SEED = {config.MAGIC_SEED}")
    
    for name, base_model in models:
        print(f"\n{name}")
        
        # Train with cross-validation
        fold_models, oof_preds, cv_rmse = train_single_model(
            base_model,
            X_train,
            y_train,
            groups,
            config.N_SPLITS,
            config.MAGIC_SEED
        )
        
        print(f"   RMSE: {cv_rmse:.4f}")
        
        # Generate test predictions (average across folds)
        test_preds = np.mean([
            model.predict(X_test) for model in fold_models
        ], axis=0)
        
        all_predictions.append(test_preds)
        model_scores[name] = cv_rmse
    
    # Weighted ensemble by inverse RMSE
    rmse_values = np.array(list(model_scores.values()))
    weights = 1 / rmse_values
    weights = weights / weights.sum()
    
    print("\n📊 Ensemble Weights:")
    for name, weight in zip(model_scores.keys(), weights):
        print(f"   {name}: {weight:.3f}")
    
    # Weighted average of predictions
    final_preds = np.average(all_predictions, axis=0, weights=weights)
    
    # Clip to non-negative (biomass can't be negative)
    final_preds = np.clip(final_preds, 0, None)
    
    return final_preds, model_scores
