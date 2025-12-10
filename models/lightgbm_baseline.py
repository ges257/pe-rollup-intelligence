"""
Train LightGBM Model for Vendor Adoption Prediction

Strategy: Manual hyperparameter tuning (start with defaults, adjust based on validation)

Hyperparameters:
- num_leaves: 31 (tree complexity)
- learning_rate: 0.05 (step size)
- scale_pos_weight: 5 (class imbalance - dev set has ~17% positives)
- feature_fraction: 0.8 (regularization)
- bagging_fraction: 0.8 (regularization)

Output:
- Trained LightGBM model
- Predictions on dev set
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import pickle


def load_feature_matrices(results_dir):
    """Load train and dev feature matrices"""
    train_df = pd.read_csv(results_dir / "tier2_feature_matrix_train.csv")
    dev_df = pd.read_csv(results_dir / "tier2_feature_matrix_dev.csv")

    return train_df, dev_df


def prepare_data(train_df, dev_df):
    """
    Prepare features and labels for LightGBM

    Returns:
    --------
    X_train, y_train, X_dev, y_dev : arrays
    feature_names : list of feature column names
    """
    # Identify feature columns (exclude identifiers and label)
    exclude_cols = ['site_id', 'vendor_id', 'label']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Extract features and labels
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values

    X_dev = dev_df[feature_cols].values
    y_dev = dev_df['label'].values

    return X_train, y_train, X_dev, y_dev, feature_cols


def train_lightgbm(X_train, y_train, X_dev, y_dev, feature_names):
    """
    Train LightGBM with manual hyperparameters

    Returns:
    --------
    model : trained LightGBM model
    """
    print("Training LightGBM...")

    # Compute class imbalance weight
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = neg_count / pos_count

    print(f"  Class distribution:")
    print(f"    Positives: {pos_count} ({pos_count/len(y_train)*100:.1f}%)")
    print(f"    Negatives: {neg_count} ({neg_count/len(y_train)*100:.1f}%)")
    print(f"    Computed pos_weight: {pos_weight:.2f}")

    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': pos_weight,
        'verbose': -1,
        'seed': 42
    }

    print(f"\n  Training parameters:")
    for key, value in params.items():
        print(f"    {key}: {value}")

    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_names
    )

    dev_data = lgb.Dataset(
        X_dev,
        label=y_dev,
        feature_name=feature_names,
        reference=train_data
    )

    # Train with early stopping
    print(f"\n  Training with early stopping...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, dev_data],
        valid_names=['train', 'dev'],
        callbacks=callbacks
    )

    print(f"\n  ✓ Training complete!")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best score (AUC): {model.best_score['dev']['auc']:.4f}")

    return model


def save_model(model, output_dir):
    """Save trained model"""
    model_path = output_dir / "tier2_lightgbm_model.txt"
    model.save_model(str(model_path))
    print(f"\n  ✓ Saved model: {model_path}")

    return model_path


def make_predictions(model, X_train, X_dev, train_df, dev_df, output_dir):
    """
    Make predictions on train and dev sets

    Returns:
    --------
    train_preds, dev_preds : arrays of predicted probabilities
    """
    print("\nMaking predictions...")

    # Predict on train
    train_preds = model.predict(X_train, num_iteration=model.best_iteration)
    print(f"  ✓ Train predictions: {len(train_preds)}")

    # Predict on dev
    dev_preds = model.predict(X_dev, num_iteration=model.best_iteration)
    print(f"  ✓ Dev predictions: {len(dev_preds)}")

    # Save predictions
    train_pred_df = train_df[['site_id', 'vendor_id', 'label']].copy()
    train_pred_df['prediction'] = train_preds
    train_pred_path = output_dir / "tier2_predictions_train.csv"
    train_pred_df.to_csv(train_pred_path, index=False)

    dev_pred_df = dev_df[['site_id', 'vendor_id', 'label']].copy()
    dev_pred_df['prediction'] = dev_preds
    dev_pred_path = output_dir / "tier2_predictions_dev.csv"
    dev_pred_df.to_csv(dev_pred_path, index=False)

    print(f"  ✓ Saved train predictions: {train_pred_path}")
    print(f"  ✓ Saved dev predictions: {dev_pred_path}")

    return train_preds, dev_preds


def main():
    """Main training pipeline"""
    print("="*70)
    print("TIER 2: TRAINING LIGHTGBM MODEL")
    print("="*70)

    # Paths
    results_dir = Path("/home/g12/pa_final_project_fall25/v0.section3.model_architecture/tier2_gradient_boosting/results")

    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    train_df, dev_df = load_feature_matrices(results_dir)
    print(f"  ✓ Train: {train_df.shape}")
    print(f"  ✓ Dev: {dev_df.shape}")

    # Prepare data
    print("\n" + "="*70)
    print("PREPARING DATA")
    print("="*70)

    X_train, y_train, X_dev, y_dev, feature_names = prepare_data(train_df, dev_df)
    print(f"  ✓ X_train: {X_train.shape}")
    print(f"  ✓ X_dev: {X_dev.shape}")
    print(f"  ✓ Features: {len(feature_names)}")

    # Train model
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)

    model = train_lightgbm(X_train, y_train, X_dev, y_dev, feature_names)

    # Save model
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)

    save_model(model, results_dir)

    # Make predictions
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)

    train_preds, dev_preds = make_predictions(
        model, X_train, X_dev, train_df, dev_df, results_dir
    )

    # Quick statistics
    print("\n" + "="*70)
    print("PREDICTION STATISTICS")
    print("="*70)

    print(f"\nTrain predictions:")
    print(f"  Mean: {train_preds.mean():.4f}")
    print(f"  Std: {train_preds.std():.4f}")
    print(f"  Min: {train_preds.min():.4f}")
    print(f"  Max: {train_preds.max():.4f}")

    print(f"\nDev predictions:")
    print(f"  Mean: {dev_preds.mean():.4f}")
    print(f"  Std: {dev_preds.std():.4f}")
    print(f"  Min: {dev_preds.min():.4f}")
    print(f"  Max: {dev_preds.max():.4f}")

    # Separation (positive vs negative)
    dev_pos = dev_preds[y_dev == 1]
    dev_neg = dev_preds[y_dev == 0]

    print(f"\nDev set separation:")
    print(f"  Positive examples mean: {dev_pos.mean():.4f}")
    print(f"  Negative examples mean: {dev_neg.mean():.4f}")
    print(f"  Difference: {dev_pos.mean() - dev_neg.mean():.4f}")

    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
