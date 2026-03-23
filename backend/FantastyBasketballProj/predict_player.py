"""
Prediction script for Fantasy Basketball CNN.
Uses the trained model to predict player performance.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from fantasy_cnn import FantasyFeatureEngine

SEQUENCE_LENGTH = 10


def predict_player(player_name, df, model, feature_engine):
    """Predict fantasy performance for a specific player."""
    player_df = df[df['player_name'] == player_name].sort_values('game_date')

    if len(player_df) == 0:
        print(f"ERROR: Player '{player_name}' not found")
        available = df['player_name'].unique()[:10]
        print(f"Available: {', '.join(available)}...")
        return None

    if len(player_df) < SEQUENCE_LENGTH:
        print(f"ERROR: {player_name} has only {len(player_df)} games (need {SEQUENCE_LENGTH})")
        return None

    last_n = player_df.tail(SEQUENCE_LENGTH)
    vecs = []
    for _, game in last_n.iterrows():
        feats = feature_engine.engineer_all_features(game.to_dict())
        vecs.append(list(feats.values()))

    X = np.array([vecs])
    preds = model.predict(X, verbose=0)

    current_avg = player_df['season_avg_fp'].iloc[-1]
    last_5_avg = player_df['last_5_avg_fp'].iloc[-1]

    return {
        'player_name': player_name,
        'current_avg': current_avg,
        'last_5_avg': last_5_avg,
        'expected_fp': float(preds[0][0][0]),
        'high_end': float(preds[1][0][0]),
        'low_end': float(preds[2][0][0]),
    }


def display_prediction(r):
    """Display prediction results."""
    trend = "UP" if r['last_5_avg'] > r['current_avg'] else "DOWN"
    print(f"\n{'=' * 50}")
    print(f"  {r['player_name']}")
    print(f"{'=' * 50}")
    print(f"  Season Avg:    {r['current_avg']:6.1f} FP")
    print(f"  Last 5 Avg:    {r['last_5_avg']:6.1f} FP  (Trending {trend})")
    print(f"  ---")
    print(f"  Predicted:     {r['expected_fp']:6.1f} FP")
    print(f"  Ceiling:       {r['high_end']:6.1f} FP")
    print(f"  Floor:         {r['low_end']:6.1f} FP")
    print(f"{'=' * 50}")


def main():
    print("FANTASY BASKETBALL CNN - PREDICTIONS\n")

    model_path = 'models/fantasy_cnn.keras'
    if not os.path.exists(model_path):
        print("ERROR: No trained model. Run train_model.py first.")
        return

    model = tf.keras.models.load_model(model_path)
    print("Model loaded.")

    data_path = 'data/player_games_2024.csv'
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    print(f"Data: {df['player_id'].nunique()} players\n")

    engine = FantasyFeatureEngine()

    players = sys.argv[1:] if len(sys.argv) > 1 else [
        "LaMelo Ball", "Shai Gilgeous-Alexander",
        "Cade Cunningham", "Anthony Edwards",
    ]

    results = []
    for name in players:
        r = predict_player(name, df, model, engine)
        if r:
            display_prediction(r)
            results.append(r)

    if results:
        print(f"\n{'Player':<25} {'Avg':>6} {'Pred':>6} {'Ceil':>6} {'Floor':>6}")
        print("-" * 55)
        results.sort(key=lambda x: x['expected_fp'], reverse=True)
        for r in results:
            print(f"{r['player_name']:<25} {r['current_avg']:6.1f} {r['expected_fp']:6.1f} "
                  f"{r['high_end']:6.1f} {r['low_end']:6.1f}")


if __name__ == '__main__':
    main()
