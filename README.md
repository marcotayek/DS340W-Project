# Context-Aware Transfer Analysis of Premier League Attackers

## Research Question
Which Premier League attackers improve after moving to another league, and which changes in context help explain that improvement?

## Overview
This project compares attacking performance before and after real transfers out of the Premier League. It combines per-90 player statistics with team, league, tactical, and playing-time context.

## Parent-paper connection
- Paper 1 motivates xG and explainable attacking metrics.
- Paper 2 motivates team context, formation, and interpretable modeling.
- Paper 3 motivates the idea that player performance changes with environment.

## Methodology
1. Identify Premier League attackers who transferred abroad.
2. Create pre-transfer and post-transfer windows.
3. Normalize performance per 90 minutes.
4. Add contextual variables such as team strength, league strength, role, formation, age, and playing time.
5. Create before/after change features.
6. Fit an interpretable Ridge/Lasso baseline.
7. Compare with Random Forest or XGBoost.
8. Explain the stronger model with feature importance or SHAP.

## Planned features
Player: goals/90, assists/90, shots/90, xG/90, xA/90, minutes, goals-minus-xG/90.
Context: team strength, league strength, possession/attacking level, role, formation, age, playing-time change.

## Current status
The project has been narrowed from general football prediction to individual transfer improvement. The pre/post framework, per-90 normalization, key attacking variables, contextual variables, and model strategy have been defined.

## Data
Add the actual raw/processed project data when redistribution is allowed. Otherwise include the exact source link and reproducible access instructions.
