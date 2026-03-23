"""
Predictions router - serves CNN model predictions from real NBA data.
Does NOT require MongoDB - works directly from CSV + trained model.
"""

import os
import sys
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

# Paths relative to where uvicorn runs (backend/)
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'FantastyBasketballProj', 'data', 'player_games_2024.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'FantastyBasketballProj', 'models', 'fantasy_cnn.keras')

# Add the ML project to path so we can import FantasyFeatureEngine
ML_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'FantastyBasketballProj')
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

# Lazy-loaded globals
_df = None
_model = None
_engine = None

SEQUENCE_LENGTH = 10


def _load_data():
    global _df
    if _df is None:
        abs_path = os.path.abspath(DATA_PATH)
        if not os.path.exists(abs_path):
            raise RuntimeError(f"Data file not found: {abs_path}")
        _df = pd.read_csv(abs_path)
    return _df


def _load_model():
    global _model
    if _model is None:
        abs_path = os.path.abspath(MODEL_PATH)
        if not os.path.exists(abs_path):
            raise RuntimeError(f"Model not found: {abs_path}")
        import tensorflow as tf
        _model = tf.keras.models.load_model(abs_path)
    return _model


def _get_engine():
    global _engine
    if _engine is None:
        from fantasy_cnn import FantasyFeatureEngine
        _engine = FantasyFeatureEngine()
    return _engine


# ---- Response models ----

class PlayerSummary(BaseModel):
    player_name: str
    team: str
    position: str
    season_avg_fp: float
    last_5_avg_fp: float
    games_played: int


class PlayerPrediction(BaseModel):
    player_name: str
    team: str
    position: str
    season_avg_fp: float
    last_5_avg_fp: float
    games_played: int
    predicted_fp: float
    ceiling: float
    floor: float


class TradeRequest(BaseModel):
    side_a: List[str]  # player names
    side_b: List[str]


class TradeSideResult(BaseModel):
    player_name: str
    predicted_fp: float
    ceiling: float
    floor: float


class TradeAnalysis(BaseModel):
    side_a: List[TradeSideResult]
    side_b: List[TradeSideResult]
    side_a_total: float
    side_b_total: float
    difference: float
    ceiling_diff: float
    floor_diff: float
    recommendation: str


# ---- Helper ----

def _predict_player(player_name: str) -> Optional[dict]:
    """Get model prediction for a player."""
    df = _load_data()
    model = _load_model()
    engine = _get_engine()

    pdf = df[df['player_name'] == player_name].sort_values('game_date')
    if len(pdf) < SEQUENCE_LENGTH:
        return None

    last_n = pdf.tail(SEQUENCE_LENGTH)
    vecs = []
    for _, row in last_n.iterrows():
        feats = engine.engineer_all_features(row.to_dict())
        vecs.append(list(feats.values()))

    X = np.array([vecs], dtype=np.float32)
    preds = model.predict(X, verbose=0)

    latest = pdf.iloc[-1]
    return {
        'player_name': player_name,
        'team': str(latest.get('team', '')),
        'position': str(latest.get('position', '')),
        'season_avg_fp': float(latest['season_avg_fp']),
        'last_5_avg_fp': float(latest['last_5_avg_fp']),
        'games_played': int(latest['games_played']),
        'predicted_fp': round(float(preds[0][0][0]), 1),
        'ceiling': round(float(preds[1][0][0]), 1),
        'floor': round(float(preds[2][0][0]), 1),
    }


# ---- Endpoints ----

@router.get("/players", response_model=List[PlayerSummary])
async def list_players(
    q: Optional[str] = Query(None, min_length=1),
    position: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """Search/list players from real NBA data (no auth required)."""
    df = _load_data()

    # Get latest row per player
    latest = df.sort_values('game_date').groupby('player_id').last().reset_index()

    if q:
        latest = latest[latest['player_name'].str.contains(q, case=False, na=False)]
    if position:
        latest = latest[latest['position'].str.contains(position, case=False, na=False)]
    if team:
        latest = latest[latest['team'].str.contains(team, case=False, na=False)]

    latest = latest.sort_values('season_avg_fp', ascending=False).head(limit)

    return [
        PlayerSummary(
            player_name=row['player_name'],
            team=str(row.get('team', '')),
            position=str(row.get('position', '')),
            season_avg_fp=round(float(row['season_avg_fp']), 1),
            last_5_avg_fp=round(float(row['last_5_avg_fp']), 1),
            games_played=int(row['games_played']),
        )
        for _, row in latest.iterrows()
    ]


@router.get("/players/positions")
async def list_positions():
    df = _load_data()
    positions = sorted(df['position'].dropna().unique().tolist())
    return {"positions": positions}


@router.get("/players/teams")
async def list_teams():
    df = _load_data()
    teams = sorted(df['team'].dropna().unique().tolist())
    return {"teams": teams}


@router.get("/predict/{player_name}", response_model=PlayerPrediction)
async def predict_player(player_name: str):
    """Get CNN prediction for a specific player."""
    result = _predict_player(player_name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found or insufficient data")
    return PlayerPrediction(**result)


@router.post("/trade", response_model=TradeAnalysis)
async def analyze_trade(request: TradeRequest):
    """Analyze a trade between two sides using CNN predictions."""
    if not request.side_a or not request.side_b:
        raise HTTPException(status_code=400, detail="Both sides must have at least one player")

    def process_side(names):
        results = []
        for name in names:
            pred = _predict_player(name)
            if pred:
                results.append(TradeSideResult(
                    player_name=pred['player_name'],
                    predicted_fp=pred['predicted_fp'],
                    ceiling=pred['ceiling'],
                    floor=pred['floor'],
                ))
            else:
                # Fallback to season average
                df = _load_data()
                pdf = df[df['player_name'] == name]
                avg = round(float(pdf['season_avg_fp'].iloc[-1]), 1) if len(pdf) > 0 else 0
                results.append(TradeSideResult(
                    player_name=name,
                    predicted_fp=avg,
                    ceiling=avg,
                    floor=avg,
                ))
        return results

    side_a = process_side(request.side_a)
    side_b = process_side(request.side_b)

    a_total = sum(p.predicted_fp for p in side_a)
    b_total = sum(p.predicted_fp for p in side_b)
    diff = b_total - a_total

    a_ceil = sum(p.ceiling for p in side_a)
    b_ceil = sum(p.ceiling for p in side_b)
    a_floor = sum(p.floor for p in side_a)
    b_floor = sum(p.floor for p in side_b)

    if diff > 3:
        rec = "ACCEPT - Side B has more value"
    elif diff < -3:
        rec = "REJECT - Side A has more value"
    else:
        rec = "NEUTRAL - Even trade"

    return TradeAnalysis(
        side_a=side_a,
        side_b=side_b,
        side_a_total=round(a_total, 1),
        side_b_total=round(b_total, 1),
        difference=round(diff, 1),
        ceiling_diff=round(b_ceil - a_ceil, 1),
        floor_diff=round(b_floor - a_floor, 1),
        recommendation=rec,
    )
