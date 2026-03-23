"""
Training script for Fantasy Basketball CNN.
Uses real NBA game log data and trains with proper targets and splits.
"""

import os
import numpy as np
import pandas as pd
from fantasy_cnn import (
    FantasyBasketballCNN,
    prepare_training_data,
    split_by_player,
)


def main():
    print("=" * 60)
    print("FANTASY BASKETBALL CNN - TRAINING")
    print("=" * 60)

    # ====================================
    # 1. LOAD DATA
    # ====================================
    data_path = 'data/player_games_2024.csv'
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run fetch_data.py first.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} game records for {df['player_id'].nunique()} players")

    # ====================================
    # 2. PREPARE SEQUENCES & TARGETS
    # ====================================
    SEQUENCE_LENGTH = 10
    LOOKAHEAD = 5

    print(f"\nBuilding sequences (window={SEQUENCE_LENGTH}, lookahead={LOOKAHEAD})...")
    X, y, player_ids = prepare_training_data(df, SEQUENCE_LENGTH, LOOKAHEAD)
    print(f"Created {len(X)} samples, input shape: {X.shape}")
    print(f"  expected_fp range: [{y['expected_fp'].min():.0f}, {y['expected_fp'].max():.0f}]")
    print(f"  high_end range:    [{y['high_end'].min():.0f}, {y['high_end'].max():.0f}]")
    print(f"  low_end range:     [{y['low_end'].min():.0f}, {y['low_end'].max():.0f}]")

    # ====================================
    # 3. SPLIT BY PLAYER (no leakage)
    # ====================================
    print("\nSplitting by player (80/20)...")
    X_train, y_train, X_val, y_val = split_by_player(X, y, player_ids, val_fraction=0.2)
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")

    # ====================================
    # 4. BUILD & TRAIN
    # ====================================
    n_features = X.shape[2]
    model = FantasyBasketballCNN(n_features=n_features, sequence_length=SEQUENCE_LENGTH)
    model.build_model()
    print(f"\nModel built with {n_features} features")
    model.model.summary()

    print("\nTraining...")
    history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    # ====================================
    # 5. EVALUATE
    # ====================================
    print("\n" + "=" * 60)
    print("VALIDATION METRICS")
    print("=" * 60)

    preds = model.predict(X_val)
    pred_expected = preds[0].flatten()
    pred_high = preds[1].flatten()
    pred_low = preds[2].flatten()

    for name, pred, actual in [
        ('expected_fp', pred_expected, y_val['expected_fp']),
        ('high_end', pred_high, y_val['high_end']),
        ('low_end', pred_low, y_val['low_end']),
    ]:
        mae = np.mean(np.abs(pred - actual))
        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        print(f"  {name:12s}  MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:6.3f}")

    # Baseline: always predict season_avg_fp
    # (grab the last game in each val sequence for its season_avg_fp)
    baseline_pred = X_val[:, -1, 2]  # column index 2 = season_avg_fp
    baseline_mae = np.mean(np.abs(baseline_pred - y_val['expected_fp']))
    print(f"\n  Baseline (season avg) MAE={baseline_mae:.2f}")
    print(f"  Model beats baseline by {baseline_mae - np.mean(np.abs(pred_expected - y_val['expected_fp'])):.2f} FP")

    # ====================================
    # 6. SAVE
    # ====================================
    os.makedirs('models', exist_ok=True)
    model.model.save('models/fantasy_cnn.keras')
    print(f"\nModel saved to models/fantasy_cnn.keras")
    print("=" * 60)


if __name__ == "__main__":
    main()
