# Football Transfer Dataset Package

This package expands the expected-goals/player-performance concepts from the first parent paper into a dataset that is directly usable for the project question:

**Which Premier League attackers improve after moving to another league, and which changes in context help explain that improvement?**

## Coverage
The builder downloads an Understat-derived public player dataset covering:
- 2014/15 through 2024/25 completed seasons
- 2025/26 current season (partial)

Big Five leagues used:
- Premier League
- La Liga
- Bundesliga
- Serie A
- Ligue 1

## What the builder creates

### `player_seasons_2014_2025.csv`
Player-club-season table with:
- goals, assists, shots
- xG, npxG, xA
- key passes
- xGChain, xGBuildup
- minutes and games
- per-90 versions
- expected attacking contribution per 90
- finishing vs xG
- team attacking-strength proxy
- league scoring-environment proxy
- partial-season flag for 2025/26

### `pl_outgoing_attacker_transfers.csv`
Main modeling table.

One row = one attacker moving from the Premier League in season t to another Big Five league in season t+1.

Includes:
- pre-transfer metrics
- post-transfer metrics
- delta/change variables
- team-context change
- league-environment change
- `delta_expected_output_p90`
- `improved_expected_output`
- current-season partial flag

### `pl_outgoing_attacker_transfers_all_candidates.csv`
Same-season EPL/non-EPL cases that need manual transfer-date verification.

## Why this is useful for the project
The original xG idea measures shot quality. This dataset extends that into transfer analysis by adding:
- per-90 normalization
- xA and chance creation
- possession-chain involvement
- before/after comparisons
- team attack context
- league scoring context
- actual observed moves rather than hypothetical transfers

## Run
From the project folder:

```bash
pip install pandas numpy
python build_dataset.py
```

The CSV files will appear in the `data/` folder.

## Important methodology note
`league_avg_team_xg_per_match` is a **scoring-environment proxy**, not an official league-quality ranking. It should be described that way in the paper/presentation.

2025/26 is still in progress and is marked as partial.
