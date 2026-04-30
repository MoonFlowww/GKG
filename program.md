# GKG Auto-Research Program

## What this project does
GKG (Generalized Knowledge Graph) maps codebases to a 3-level graph
(MACRO→MESO→MICRO) and uses it to guide LLM code navigation.
We compare three conditions: baseline_full, baseline_nav, gkg_nav.

## Fixed — never modify
- gkg.py, gkg_navigator.py, ast_mapper.py, ollama_client.py
- ab_quests.py, ab_stats.py, gkg_viz.py

## Mutable — your workspace
- ab_runner.py (system prompts, caps, heuristics)

## Metric (single number, higher is better)
SCORE/TOKENS*1000 — printed at the end of run_eval.py as EFFICIENCY=...
Secondary: SCORE alone (quality without token penalty)

## Experiment loop
1. Read results.tsv to see current best EFFICIENCY
2. Propose ONE change to ab_runner.py
3. Run: python run_eval.py "description of change"
4. Read the new row in results.tsv
5. If new EFFICIENCY > best: git add ab_runner.py && git commit -m "improvement: <description>"
6. Else: git checkout ab_runner.py
7. Repeat

## Research directions (priority order)
1. _GKG_NAV_SYS prompt — Q5 (analysis) scores 0%. Fix it.
2. _auto_route keyword extraction — better keywords → fewer wasted turns
3. MAX_CTX_CHARS threshold — currently 14000, try tuning
4. _GEN_NUDGE wording — does it help or add noise?
5. MAX_FETCHES cap — currently 5, try 3 or 7

## Constraints
- ONE change per experiment
- Do not change the eval quests or scoring
- Do not install new packages
- Each run takes ~15 min with the 1.5b model — plan accordingly
