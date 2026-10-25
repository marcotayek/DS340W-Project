"""
Build a context-aware football transfer dataset from Understat-derived player-season data.

Outputs
-------
1. data/player_seasons_2014_2025.csv
   Club-level player-season records with per-90 and contextual features.

2. data/pl_outgoing_attacker_transfers.csv
   Reliable season-to-season transfers where a player was in the EPL in season t
   and in a non-EPL league in season t+1, with pre/post metrics and deltas.

3. data/pl_outgoing_attacker_transfers_all_candidates.csv
   Broader candidate table, including ambiguous same-season multi-club cases for
   manual review.

Data source
-----------
https://github.com/vibedatascience/understat_players_aggregated
The combined source file covers 2014/15 through the current 2025/26 season.
2025/26 is partial while the season is in progress.
"""

from pathlib import Path
import numpy as np
import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/vibedatascience/"
    "understat_players_aggregated/main/"
    "understat_players_aggregated_2014_td.csv"
)

OUT = Path("data")
OUT.mkdir(exist_ok=True)

TOP5 = {"EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"}
CURRENT_YEAR = 2025

# Keep forwards by default. Set INCLUDE_MIDFIELDERS=True if you want attacking
# midfielders/wide midfielders included in the initial candidate pool.
INCLUDE_MIDFIELDERS = False
MIN_PRE_MINUTES = 450
MIN_POST_MINUTES = 450


def safe_per90(series, minutes):
    minutes = pd.to_numeric(minutes, errors="coerce")
    series = pd.to_numeric(series, errors="coerce")
    return np.where(minutes > 0, series * 90.0 / minutes, np.nan)


def load_source():
    df = pd.read_csv(SOURCE_URL)
    # Keep Big Five for cleaner league-to-league comparison.
    df = df[df["league"].isin(TOP5)].copy()

    numeric_cols = [
        "id", "year", "games", "time", "goals", "xG", "npg", "npxG",
        "assists", "xA", "shots", "key_passes", "xGChain", "xGBuildup",
        "yellow_cards", "red_cards"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["season_start"] = df["year"].astype("Int64")
    df["season_label"] = (
        df["season_start"].astype(str)
        + "-"
        + ((df["season_start"] + 1) % 100).astype(str).str.zfill(2)
    )
    df["is_partial_season"] = df["season_start"].eq(CURRENT_YEAR)
    return df


def add_player_metrics(df):
    df = df.copy()
    metric_map = {
        "goals": "goals_p90",
        "xG": "xg_p90",
        "npg": "npg_p90",
        "npxG": "npxg_p90",
        "assists": "assists_p90",
        "xA": "xa_p90",
        "shots": "shots_p90",
        "key_passes": "key_passes_p90",
        "xGChain": "xgchain_p90",
        "xGBuildup": "xgbuildup_p90",
    }
    for raw, new in metric_map.items():
        df[new] = safe_per90(df[raw], df["time"])

    df["goal_contrib_p90"] = df["goals_p90"] + df["assists_p90"]
    df["expected_contrib_p90"] = df["xg_p90"] + df["xa_p90"]
    df["finishing_vs_xg_p90"] = df["goals_p90"] - df["xg_p90"]
    df["npg_vs_npxg_p90"] = df["npg_p90"] - df["npxg_p90"]

    # Useful proxy for involvement in dangerous possession chains.
    df["creative_involvement_p90"] = (
        df["xa_p90"].fillna(0)
        + df["key_passes_p90"].fillna(0)
        + df["xgchain_p90"].fillna(0)
    )
    return df


def add_team_and_league_context(df):
    """
    Context is intentionally computed only from columns available in the
    Understat-derived source, so the dataset stays reproducible.

    team_xg_per_match:
        Sum of player xG for the team / approximate team matches.
        max(games) is used as a practical estimate of league matches represented.

    team_attack_strength_index:
        Team xG/match divided by league-season average team xG/match.
        >1 = stronger-than-average attack in that league-season.

    league_avg_team_xg_per_match:
        Scoring/chance environment proxy, NOT an absolute UEFA-quality ranking.
    """
    df = df.copy()

    team = (
        df.groupby(["league", "season_start", "team_title"], as_index=False)
        .agg(
            team_total_xg=("xG", "sum"),
            team_total_goals=("goals", "sum"),
            approx_team_matches=("games", "max"),
        )
    )
    team["team_xg_per_match"] = (
        team["team_total_xg"] / team["approx_team_matches"].replace(0, np.nan)
    )
    team["team_goals_per_match"] = (
        team["team_total_goals"] / team["approx_team_matches"].replace(0, np.nan)
    )

    league = (
        team.groupby(["league", "season_start"], as_index=False)
        .agg(
            league_avg_team_xg_per_match=("team_xg_per_match", "mean"),
            league_avg_team_goals_per_match=("team_goals_per_match", "mean"),
        )
    )

    team = team.merge(league, on=["league", "season_start"], how="left")
    team["team_attack_strength_index"] = (
        team["team_xg_per_match"]
        / team["league_avg_team_xg_per_match"].replace(0, np.nan)
    )

    df = df.merge(
        team[
            [
                "league", "season_start", "team_title",
                "team_xg_per_match", "team_goals_per_match",
                "league_avg_team_xg_per_match",
                "league_avg_team_goals_per_match",
                "team_attack_strength_index",
            ]
        ],
        on=["league", "season_start", "team_title"],
        how="left",
    )
    return df


def attacker_mask(df):
    # Understat aggregate source provides primary_position F/M/D/GK/S.
    mask = df["primary_position"].eq("F")
    if INCLUDE_MIDFIELDERS:
        mask = mask | df["primary_position"].eq("M")
    return mask


def make_reliable_transfer_pairs(df):
    """
    Reliable summer/off-season transitions:
      EPL in season t -> non-EPL in season t+1
    We use the club row with the most minutes if a player has multiple clubs
    in the same league-season.
    """
    attackers = df[attacker_mask(df)].copy()

    # Pick main club in each league-season for a player by minutes.
    main = (
        attackers.sort_values(["id", "season_start", "league", "time"])
        .groupby(["id", "season_start", "league"], as_index=False)
        .tail(1)
        .copy()
    )

    pre = main[main["league"].eq("EPL")].copy()
    post = main[~main["league"].eq("EPL")].copy()

    pre["join_year"] = pre["season_start"] + 1
    pairs = pre.merge(
        post,
        left_on=["id", "join_year"],
        right_on=["id", "season_start"],
        suffixes=("_pre", "_post"),
        how="inner",
    )

    # Remove cases where player also has EPL data in the post season,
    # because these may be loans/partial seasons or ambiguous direction.
    epl_post_keys = set(
        map(tuple, main[main["league"].eq("EPL")][["id", "season_start"]].to_numpy())
    )
    pairs["has_epl_record_in_post_season"] = [
        (pid, yr) in epl_post_keys
        for pid, yr in zip(pairs["id"], pairs["season_start_post"])
    ]
    pairs = pairs[~pairs["has_epl_record_in_post_season"]].copy()

    pairs = pairs[
        (pairs["time_pre"] >= MIN_PRE_MINUTES)
        & (pairs["time_post"] >= MIN_POST_MINUTES)
    ].copy()

    metrics = [
        "goals_p90", "assists_p90", "goal_contrib_p90",
        "xg_p90", "xa_p90", "expected_contrib_p90",
        "shots_p90", "key_passes_p90",
        "xgchain_p90", "xgbuildup_p90",
        "finishing_vs_xg_p90",
        "team_xg_per_match", "team_goals_per_match",
        "team_attack_strength_index",
        "league_avg_team_xg_per_match",
        "league_avg_team_goals_per_match",
    ]
    for m in metrics:
        pairs[f"delta_{m}"] = pairs[f"{m}_post"] - pairs[f"{m}_pre"]

    # Stable project targets: keep several rather than forcing one arbitrary label.
    pairs["delta_attacking_output_p90"] = (
        pairs["goal_contrib_p90_post"] - pairs["goal_contrib_p90_pre"]
    )
    pairs["delta_expected_output_p90"] = (
        pairs["expected_contrib_p90_post"] - pairs["expected_contrib_p90_pre"]
    )

    # Simple interpretable classification for exploratory work.
    # Positive expected-output change = improved underlying attacking output.
    pairs["improved_expected_output"] = (
        pairs["delta_expected_output_p90"] > 0
    ).astype(int)

    pairs["transfer_type"] = "EPL_to_other_league_next_season"
    pairs["direction_confidence"] = "high"
    pairs["post_season_partial"] = pairs["season_start_post"].eq(CURRENT_YEAR)

    keep = [
        "id",
        "player_name_pre",
        "season_label_pre", "team_title_pre", "league_pre",
        "season_label_post", "team_title_post", "league_post",
        "time_pre", "time_post",
        "position_pre", "position_post",
        "primary_position_pre", "primary_position_post",
        "is_partial_season_pre", "is_partial_season_post",
        "goals_p90_pre", "goals_p90_post", "delta_goals_p90",
        "assists_p90_pre", "assists_p90_post", "delta_assists_p90",
        "goal_contrib_p90_pre", "goal_contrib_p90_post",
        "delta_goal_contrib_p90",
        "xg_p90_pre", "xg_p90_post", "delta_xg_p90",
        "xa_p90_pre", "xa_p90_post", "delta_xa_p90",
        "expected_contrib_p90_pre", "expected_contrib_p90_post",
        "delta_expected_contrib_p90",
        "shots_p90_pre", "shots_p90_post", "delta_shots_p90",
        "key_passes_p90_pre", "key_passes_p90_post",
        "delta_key_passes_p90",
        "xgchain_p90_pre", "xgchain_p90_post", "delta_xgchain_p90",
        "xgbuildup_p90_pre", "xgbuildup_p90_post", "delta_xgbuildup_p90",
        "finishing_vs_xg_p90_pre", "finishing_vs_xg_p90_post",
        "delta_finishing_vs_xg_p90",
        "team_xg_per_match_pre", "team_xg_per_match_post",
        "delta_team_xg_per_match",
        "team_attack_strength_index_pre", "team_attack_strength_index_post",
        "delta_team_attack_strength_index",
        "league_avg_team_xg_per_match_pre",
        "league_avg_team_xg_per_match_post",
        "delta_league_avg_team_xg_per_match",
        "delta_attacking_output_p90",
        "delta_expected_output_p90",
        "improved_expected_output",
        "transfer_type", "direction_confidence",
        "post_season_partial",
    ]
    return pairs[[c for c in keep if c in pairs.columns]].sort_values(
        ["season_label_pre", "player_name_pre"]
    )


def make_same_season_candidates(df):
    """
    Candidate list for manual review of mid-season transfers.
    Understat's season aggregate tells us that the player appeared for clubs in
    both EPL and another league in the same season, but not the exact chronology.
    These should not be treated as confirmed EPL-outgoing transfers without a
    transfer-date source.
    """
    attackers = df[attacker_mask(df)].copy()
    league_counts = (
        attackers.groupby(["id", "season_start"])["league"]
        .agg(lambda x: sorted(set(x)))
        .reset_index(name="leagues_seen")
    )
    cand = league_counts[
        league_counts["leagues_seen"].apply(
            lambda xs: "EPL" in xs and any(x != "EPL" for x in xs)
        )
    ].copy()
    cand["direction_confidence"] = "manual_review_required"
    cand["note"] = (
        "Player has EPL and non-EPL records in same season; add transfer-date "
        "data before treating direction as confirmed."
    )
    return cand


def main():
    raw = load_source()
    enriched = add_player_metrics(raw)
    enriched = add_team_and_league_context(enriched)

    # Save the useful club-level source for modeling.
    player_cols = [
        "id", "player_name", "team_title", "league",
        "season_start", "season_label", "is_partial_season",
        "position", "primary_position", "games", "time",
        "goals", "xG", "npg", "npxG", "assists", "xA", "shots",
        "key_passes", "xGChain", "xGBuildup",
        "goals_p90", "xg_p90", "npg_p90", "npxg_p90",
        "assists_p90", "xa_p90", "shots_p90", "key_passes_p90",
        "xgchain_p90", "xgbuildup_p90",
        "goal_contrib_p90", "expected_contrib_p90",
        "finishing_vs_xg_p90", "npg_vs_npxg_p90",
        "creative_involvement_p90",
        "team_xg_per_match", "team_goals_per_match",
        "team_attack_strength_index",
        "league_avg_team_xg_per_match",
        "league_avg_team_goals_per_match",
    ]
    enriched[[c for c in player_cols if c in enriched.columns]].to_csv(
        OUT / "player_seasons_2014_2025.csv", index=False
    )

    pairs = make_reliable_transfer_pairs(enriched)
    pairs.to_csv(OUT / "pl_outgoing_attacker_transfers.csv", index=False)

    same_season = make_same_season_candidates(enriched)
    same_season.to_csv(
        OUT / "pl_outgoing_attacker_transfers_all_candidates.csv", index=False
    )

    print(f"Saved {len(enriched):,} player-club-season rows.")
    print(f"Saved {len(pairs):,} high-confidence EPL-outgoing transfer pairs.")
    print(f"Saved {len(same_season):,} same-season candidates for manual review.")
    print("NOTE: 2025/26 is partial and is marked with is_partial_season/post_season_partial.")


if __name__ == "__main__":
    main()
