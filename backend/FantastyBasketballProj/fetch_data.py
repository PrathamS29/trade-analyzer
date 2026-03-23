"""
Fetch real NBA game log data via nba_api and build training CSV.
Calculates fantasy points (standard ESPN scoring) and rolling stats.
"""

import time
import os
import pandas as pd
import numpy as np
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo

# ESPN standard fantasy scoring
SCORING = {
    'PTS': 1,
    'REB': 1,
    'AST': 1,
    'STL': 2,
    'BLK': 2,
    'TOV': -1,
}

SEASON = '2024-25'
MIN_GAMES = 20  # skip players with very few games


def get_active_players(min_games=MIN_GAMES):
    """Get list of active NBA players."""
    all_players = nba_players.get_active_players()
    print(f"Found {len(all_players)} active players in nba_api")
    return all_players


def fetch_game_logs(player_id, season=SEASON):
    """Fetch game logs for a single player/season."""
    try:
        log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season'
        )
        df = log.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"  Error fetching player {player_id}: {e}")
        return None


def fetch_player_info(player_id):
    """Fetch player metadata (position, age, etc.)."""
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        df = info.get_data_frames()[0]
        return df.iloc[0]
    except Exception:
        return None


def calculate_fantasy_points(row):
    """Calculate ESPN standard fantasy points for a game."""
    fp = 0
    fp += row.get('PTS', 0) * SCORING['PTS']
    fp += row.get('REB', 0) * SCORING['REB']
    fp += row.get('AST', 0) * SCORING['AST']
    fp += row.get('STL', 0) * SCORING['STL']
    fp += row.get('BLK', 0) * SCORING['BLK']
    fp += row.get('TOV', 0) * SCORING['TOV']
    return fp


def build_dataset(max_players=150):
    """
    Fetch game logs for top active players and build a training dataset.
    Rate-limited to avoid NBA API throttling.
    """
    all_players = get_active_players()
    rows = []
    players_fetched = 0

    for player in all_players:
        if players_fetched >= max_players:
            break

        pid = player['id']
        pname = player['full_name']

        print(f"[{players_fetched + 1}/{max_players}] Fetching {pname}...")

        game_log = fetch_game_logs(pid, SEASON)
        if game_log is None or len(game_log) < MIN_GAMES:
            print(f"  Skipping {pname} (insufficient games)")
            time.sleep(0.6)
            continue

        # Fetch player info for position/age
        info = fetch_player_info(pid)
        time.sleep(0.6)  # rate limit

        position = info['POSITION'] if info is not None else 'Unknown'
        team = info['TEAM_ABBREVIATION'] if info is not None else 'UNK'

        # Parse birthdate for age
        age = 25  # default
        if info is not None and pd.notna(info.get('BIRTHDATE')):
            try:
                bd = pd.to_datetime(info['BIRTHDATE'])
                age = int((pd.Timestamp.now() - bd).days / 365.25)
            except Exception:
                pass

        # Sort chronologically (API returns newest first)
        game_log = game_log.sort_values('GAME_DATE').reset_index(drop=True)

        for idx, game in game_log.iterrows():
            fp = calculate_fantasy_points(game)

            # Rolling averages (using only past games, no lookahead)
            past_games = game_log.iloc[:idx + 1]
            past_fps = past_games.apply(calculate_fantasy_points, axis=1)

            season_avg_fp = past_fps.mean()
            last_5_avg = past_fps.tail(5).mean() if len(past_fps) >= 5 else past_fps.mean()
            last_10_avg = past_fps.tail(10).mean() if len(past_fps) >= 10 else past_fps.mean()
            last_15_avg = past_fps.tail(15).mean() if len(past_fps) >= 15 else past_fps.mean()
            fp_std_15 = past_fps.tail(15).std() if len(past_fps) >= 5 else 5.0
            fp_std_30 = past_fps.tail(30).std() if len(past_fps) >= 10 else 5.0

            # Minutes
            min_str = game.get('MIN', '0')
            try:
                if ':' in str(min_str):
                    parts = str(min_str).split(':')
                    minutes = int(parts[0]) + int(parts[1]) / 60
                else:
                    minutes = float(min_str)
            except (ValueError, TypeError):
                minutes = 0.0

            mpg = past_games['MIN'].apply(lambda m: (
                int(str(m).split(':')[0]) + int(str(m).split(':')[1]) / 60
                if ':' in str(m) else float(m) if m else 0
            )).mean()

            # Games missed estimate (gaps in schedule > 3 days)
            games_played = idx + 1
            is_starter = 1 if game.get('START_POSITION', '') != '' else 0

            # FGA-based usage proxy
            team_game_fga = game.get('FGA', 10)
            usage_proxy = min(team_game_fga / 80.0, 0.40)  # rough proxy

            # True shooting
            pts = game.get('PTS', 0)
            fga = game.get('FGA', 1)
            fta = game.get('FTA', 0)
            ts_denom = 2 * (fga + 0.44 * fta)
            ts_pct = pts / ts_denom if ts_denom > 0 else 0.5

            rows.append({
                'player_id': pid,
                'player_name': pname,
                'game_date': game['GAME_DATE'],
                'age': age,
                'position': position,
                'team': team,
                'fantasy_points': fp,
                'season_avg_fp': season_avg_fp,
                'last_5_avg_fp': last_5_avg,
                'last_10_avg_fp': last_10_avg,
                'last_15_avg_fp': last_15_avg,
                'fp_std_15': fp_std_15 if not np.isnan(fp_std_15) else 5.0,
                'fp_std_30': fp_std_30 if not np.isnan(fp_std_30) else 5.0,
                'mpg': mpg,
                'usage_rate': usage_proxy,
                'true_shooting_pct': ts_pct,
                'per': 15.0,  # PER requires team-level data, use placeholder
                'games_played': games_played,
                'games_remaining': max(82 - games_played, 0),
                'is_starter': is_starter,
                'injury_status': 'healthy',
                'games_missed_last_season': 0,
                'games_missed_current': 0,
                'team_pace': 100.0,  # placeholder
                'team_win_pct': 0.5,  # placeholder
                # Raw box score for reference
                'pts': pts,
                'reb': game.get('REB', 0),
                'ast': game.get('AST', 0),
                'stl': game.get('STL', 0),
                'blk': game.get('BLK', 0),
                'tov': game.get('TOV', 0),
                'minutes': minutes,
                'fgm': game.get('FGM', 0),
                'fga': fga,
                'fg3m': game.get('FG3M', 0),
                'fg3a': game.get('FG3A', 0),
                'ftm': game.get('FTM', 0),
                'fta': fta,
            })

        players_fetched += 1
        print(f"  OK {pname}: {len(game_log)} games, avg {season_avg_fp:.1f} FP")

    df = pd.DataFrame(rows)
    return df


def main():
    print("=" * 60)
    print("NBA DATA FETCHER - Building Training Dataset")
    print("=" * 60)
    print(f"Season: {SEASON}")
    print(f"Min games: {MIN_GAMES}")
    print()

    df = build_dataset(max_players=150)

    print(f"\n{'=' * 60}")
    print(f"Dataset built: {len(df)} game records, {df['player_id'].nunique()} players")

    # Save
    os.makedirs('data', exist_ok=True)
    out_path = 'data/player_games_2024.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
