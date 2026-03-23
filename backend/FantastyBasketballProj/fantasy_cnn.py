"""
Fantasy Basketball CNN Trade Analyzer
Complete implementation with feature engineering and model architecture.
Designed to work with real NBA game log data from nba_api.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional


# ============================================================================
# PART 1: FEATURE ENGINEERING PIPELINE
# ============================================================================

# Feature names produced by engineer_all_features (order matters for model input)
FEATURE_NAMES: List[str] = []  # populated on first call


class FantasyFeatureEngine:
    """
    Feature engineering grounded in columns available from fetch_data.py.
    Every feature is derived from real box-score data.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Derived risk / context scores (use only available columns)
    # ------------------------------------------------------------------

    def calculate_injury_risk_score(self, player_data: Dict) -> float:
        """Injury risk score (0-1 scale) based on age and games missed."""
        age = player_data.get('age', 25)
        games_missed_last = player_data.get('games_missed_last_season', 0)
        games_missed_current = player_data.get('games_missed_current', 0)
        games_played = player_data.get('games_played', 1)

        if age < 27:
            age_factor = 0.0
        elif age <= 30:
            age_factor = 0.1
        elif age <= 33:
            age_factor = 0.3
        else:
            age_factor = 0.5

        injury_risk = (
            (games_missed_last / 82) * 0.3
            + (games_missed_current / max(games_played, 1)) * 0.4
            + age_factor * 0.3
        )
        return min(injury_risk, 1.0)

    def calculate_trend_score(self, player_data: Dict) -> float:
        """How much the player is trending up or down vs season average.
        Positive = hot streak, negative = cold streak. Normalised by avg."""
        season_avg = player_data.get('season_avg_fp', 1)
        last_5 = player_data.get('last_5_avg_fp', season_avg)
        if season_avg == 0:
            return 0.0
        return (last_5 - season_avg) / max(season_avg, 1)

    def calculate_consistency_score(self, player_data: Dict) -> float:
        """Lower std-dev relative to mean = more consistent (0-1, 1 = very consistent)."""
        std = player_data.get('fp_std_15', 5)
        avg = player_data.get('season_avg_fp', 1)
        cv = std / max(avg, 1)
        return max(1.0 - cv, 0.0)

    # ------------------------------------------------------------------
    # Main feature extraction
    # ------------------------------------------------------------------

    def engineer_all_features(self, player_data: Dict) -> Dict:
        """
        Extract all features from a single game row.
        Returns an ordered dict whose values become model input.
        """
        f = {}

        # --- Basic stats ---
        f['age'] = player_data.get('age', 25)
        f['games_played'] = player_data.get('games_played', 0)
        f['season_avg_fp'] = player_data.get('season_avg_fp', 0)
        f['minutes'] = player_data.get('minutes', player_data.get('mpg', 0))
        f['mpg'] = player_data.get('mpg', 0)
        f['usage_rate'] = player_data.get('usage_rate', 0.2)
        f['true_shooting_pct'] = player_data.get('true_shooting_pct', 0.5)
        f['is_starter'] = float(player_data.get('is_starter', 0))

        # --- Box score (this game) ---
        f['pts'] = player_data.get('pts', 0)
        f['reb'] = player_data.get('reb', 0)
        f['ast'] = player_data.get('ast', 0)
        f['stl'] = player_data.get('stl', 0)
        f['blk'] = player_data.get('blk', 0)
        f['tov'] = player_data.get('tov', 0)
        f['fgm'] = player_data.get('fgm', 0)
        f['fga'] = player_data.get('fga', 0)
        f['fg3m'] = player_data.get('fg3m', 0)
        f['fg3a'] = player_data.get('fg3a', 0)
        f['ftm'] = player_data.get('ftm', 0)
        f['fta'] = player_data.get('fta', 0)

        # --- Rolling averages / trends ---
        f['last_5_avg_fp'] = player_data.get('last_5_avg_fp', 0)
        f['last_10_avg_fp'] = player_data.get('last_10_avg_fp', 0)
        f['last_15_avg_fp'] = player_data.get('last_15_avg_fp', 0)
        f['fp_std_15'] = player_data.get('fp_std_15', 5)
        f['fp_std_30'] = player_data.get('fp_std_30', 5)

        # --- Derived scores ---
        f['injury_risk'] = self.calculate_injury_risk_score(player_data)
        f['trend_score'] = self.calculate_trend_score(player_data)
        f['consistency'] = self.calculate_consistency_score(player_data)

        # --- Position one-hot ---
        position = str(player_data.get('position', ''))
        for pos in ['Guard', 'Forward', 'Center']:
            f[f'pos_{pos}'] = float(pos in position)

        # Cache feature names on first call
        global FEATURE_NAMES
        if not FEATURE_NAMES:
            FEATURE_NAMES = list(f.keys())

        return f

    # ------------------------------------------------------------------
    # Simple heuristic projections (used by TradeAnalyzer when no model)
    # ------------------------------------------------------------------

    def calculate_high_low_projections(self, features: Dict) -> Tuple[float, float, float]:
        """Heuristic ceiling / floor / expected projection."""
        base = features.get('season_avg_fp', 0)
        std = features.get('fp_std_15', 5)
        trend = features.get('trend_score', 0)

        expected = base * 0.7 + features.get('last_5_avg_fp', base) * 0.3
        high = expected + std * 1.0 + base * max(trend, 0) * 0.1
        low = expected - std * 1.0 + base * min(trend, 0) * 0.1
        low -= base * features.get('injury_risk', 0) * 0.1

        return max(high, 0), max(low, 0), max(expected, 0)


# ============================================================================
# PART 2: CNN MODEL ARCHITECTURE
# ============================================================================

class FantasyBasketballCNN:
    """
    Hybrid CNN-LSTM model with multi-head attention for fantasy predictions.

    Output heads
    -------------
    expected_fp : predicted fantasy points for the next game
    high_end    : 90th-percentile outcome over next 5 games
    low_end     : 10th-percentile outcome over next 5 games
    """

    def __init__(self, n_features: int = 32, sequence_length: int = 10):
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.model = None
        self.feature_engine = FantasyFeatureEngine()

    def build_model(self):
        """Build the complete model architecture."""
        inp = layers.Input(shape=(self.sequence_length, self.n_features))

        # ===== Branch 1: Conv1D for local pattern recognition =====
        c = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inp)
        c = layers.BatchNormalization()(c)
        c = layers.Dropout(0.2)(c)

        c = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(c)
        c = layers.BatchNormalization()(c)
        c = layers.Dropout(0.3)(c)

        c = layers.Conv1D(64, kernel_size=2, activation='relu', padding='same')(c)
        c = layers.BatchNormalization()(c)

        # Multi-head attention on conv output — lets the model learn which
        # games in the window are most important for predicting the next one.
        att = layers.MultiHeadAttention(num_heads=4, key_dim=16)(c, c)
        att = layers.GlobalAveragePooling1D()(att)

        conv_flat = layers.Flatten()(c)

        # ===== Branch 2: LSTM for temporal dependencies =====
        l = layers.LSTM(64, return_sequences=True)(inp)
        l = layers.Dropout(0.3)(l)
        l = layers.LSTM(32, return_sequences=False)(l)
        l = layers.Dropout(0.3)(l)

        # ===== Merge =====
        merged = layers.Concatenate()([conv_flat, l, att])

        x = layers.Dense(128, activation='relu')(merged)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)

        # ===== Output heads =====
        out_expected = layers.Dense(1, name='expected_fp')(x)
        out_high = layers.Dense(1, name='high_end')(x)
        out_low = layers.Dense(1, name='low_end')(x)

        self.model = models.Model(
            inputs=inp,
            outputs=[out_expected, out_high, out_low],
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'expected_fp': 'mse',
                'high_end': 'mse',
                'low_end': 'mse',
            },
            loss_weights={
                'expected_fp': 2.0,
                'high_end': 1.0,
                'low_end': 1.0,
            },
            metrics={'expected_fp': 'mae', 'high_end': 'mae', 'low_end': 'mae'},
        )
        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model with early stopping and LR reduction."""
        callbacks = [
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4),
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
        )
        return history

    def predict(self, X):
        """Make predictions. Returns (expected, high, low) arrays."""
        return self.model.predict(X, verbose=0)


# ============================================================================
# PART 3: TRADE ANALYZER
# ============================================================================

class TradeAnalyzer:
    """
    Analyze fantasy trades using model predictions or heuristic fallback.
    """

    def __init__(self, model: Optional[FantasyBasketballCNN] = None,
                 player_df: Optional[pd.DataFrame] = None):
        self.model = model
        self.feature_engine = FantasyFeatureEngine()
        self.player_df = player_df  # full game log dataframe

    def _get_player_prediction(self, player_name: str) -> Dict:
        """Get model prediction for a player using their last N games."""
        if self.player_df is None or self.model is None:
            return {}

        pdf = self.player_df[self.player_df['player_name'] == player_name]
        if len(pdf) == 0:
            return {}

        pdf = pdf.sort_values('game_date')
        seq_len = self.model.sequence_length

        if len(pdf) < seq_len:
            return {}

        last_n = pdf.tail(seq_len)
        vecs = []
        for _, row in last_n.iterrows():
            feats = self.feature_engine.engineer_all_features(row.to_dict())
            vecs.append(list(feats.values()))

        X = np.array([vecs])
        preds = self.model.predict(X)
        return {
            'expected_fp': float(preds[0][0][0]),
            'high_end': float(preds[1][0][0]),
            'low_end': float(preds[2][0][0]),
        }

    def analyze_trade(self,
                      giving_names: List[str],
                      receiving_names: List[str]) -> Dict:
        """Analyze a trade between two sides of player names."""
        def side_value(names):
            total_exp, total_high, total_low = 0, 0, 0
            details = []
            for name in names:
                pred = self._get_player_prediction(name)
                if pred:
                    total_exp += pred['expected_fp']
                    total_high += pred['high_end']
                    total_low += pred['low_end']
                    details.append({'name': name, **pred})
                else:
                    # Fallback: use season average from dataframe
                    pdf = self.player_df[self.player_df['player_name'] == name]
                    if len(pdf) > 0:
                        avg = pdf['season_avg_fp'].iloc[-1]
                    else:
                        avg = 0
                    total_exp += avg
                    total_high += avg
                    total_low += avg
                    details.append({'name': name, 'expected_fp': avg,
                                    'high_end': avg, 'low_end': avg})
            return total_exp, total_high, total_low, details

        give_exp, give_high, give_low, give_details = side_value(giving_names)
        recv_exp, recv_high, recv_low, recv_details = side_value(receiving_names)

        diff = recv_exp - give_exp

        if diff > 3:
            rec = 'ACCEPT'
        elif diff < -3:
            rec = 'REJECT'
        else:
            rec = 'NEUTRAL'

        return {
            'giving': give_details,
            'receiving': recv_details,
            'giving_total': give_exp,
            'receiving_total': recv_exp,
            'value_difference': diff,
            'ceiling_comparison': recv_high - give_high,
            'floor_comparison': recv_low - give_low,
            'recommendation': rec,
        }


# ============================================================================
# PART 4: DATA PREPARATION UTILITIES
# ============================================================================

def prepare_training_data(df: pd.DataFrame,
                          sequence_length: int = 10,
                          lookahead: int = 5) -> Tuple[np.ndarray, Dict, List[int]]:
    """
    Build (X, y) arrays from a game-log DataFrame.

    Targets (derived from real game outcomes, NOT a formula):
      expected_fp : actual fantasy points of the very next game
      high_end    : 90th percentile of the next `lookahead` games
      low_end     : 10th percentile of the next `lookahead` games

    Returns:
        X            : (n_samples, sequence_length, n_features)
        y            : dict of target arrays
        player_ids   : list of player_id per sample (for splitting)
    """
    engine = FantasyFeatureEngine()

    X_data = []
    y_expected = []
    y_high = []
    y_low = []
    player_ids = []

    for pid in df['player_id'].unique():
        pdf = df[df['player_id'] == pid].sort_values('game_date').reset_index(drop=True)

        # Need sequence_length games for input + lookahead games for targets
        if len(pdf) < sequence_length + lookahead:
            continue

        for i in range(len(pdf) - sequence_length - lookahead + 1):
            seq = pdf.iloc[i:i + sequence_length]
            future = pdf.iloc[i + sequence_length:i + sequence_length + lookahead]

            # Features for each game in the window
            vecs = []
            for _, game in seq.iterrows():
                feats = engine.engineer_all_features(game.to_dict())
                vecs.append(list(feats.values()))

            X_data.append(vecs)

            # Targets from real outcomes
            next_fp = pdf.iloc[i + sequence_length]['fantasy_points']
            future_fps = future['fantasy_points'].values

            y_expected.append(float(next_fp))
            y_high.append(float(np.percentile(future_fps, 90)))
            y_low.append(float(np.percentile(future_fps, 10)))
            player_ids.append(pid)

    X = np.array(X_data, dtype=np.float32)
    y = {
        'expected_fp': np.array(y_expected, dtype=np.float32),
        'high_end': np.array(y_high, dtype=np.float32),
        'low_end': np.array(y_low, dtype=np.float32),
    }
    return X, y, player_ids


def split_by_player(X, y, player_ids, val_fraction=0.2):
    """
    Split data by player to prevent leakage.
    All games from a given player go into either train or val, never both.
    """
    unique_pids = list(set(player_ids))
    np.random.seed(42)
    np.random.shuffle(unique_pids)
    split = int(len(unique_pids) * (1 - val_fraction))
    train_pids = set(unique_pids[:split])

    train_mask = np.array([pid in train_pids for pid in player_ids])
    val_mask = ~train_mask

    X_train = X[train_mask]
    X_val = X[val_mask]
    y_train = {k: v[train_mask] for k, v in y.items()}
    y_val = {k: v[val_mask] for k, v in y.items()}

    return X_train, y_train, X_val, y_val


# ============================================================================
# PART 5: QUICK DEMO
# ============================================================================

if __name__ == "__main__":
    engine = FantasyFeatureEngine()
    sample = {
        'player_name': 'LaMelo Ball', 'age': 24, 'games_played': 30,
        'season_avg_fp': 44.7, 'last_5_avg_fp': 48.2, 'last_10_avg_fp': 46.5,
        'last_15_avg_fp': 45.1, 'fp_std_15': 9.8, 'fp_std_30': 10.2,
        'mpg': 34.5, 'usage_rate': 0.29, 'true_shooting_pct': 0.545,
        'pts': 28, 'reb': 6, 'ast': 8, 'stl': 1, 'blk': 0, 'tov': 3,
        'fgm': 10, 'fga': 22, 'fg3m': 3, 'fg3a': 8, 'ftm': 5, 'fta': 6,
        'minutes': 34, 'is_starter': 1, 'position': 'Guard',
    }
    feats = engine.engineer_all_features(sample)
    high, low, exp = engine.calculate_high_low_projections(feats)
    print(f"Features: {len(feats)}")
    print(f"Expected: {exp:.1f}  Ceiling: {high:.1f}  Floor: {low:.1f}")
