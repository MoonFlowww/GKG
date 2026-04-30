"""Single eval run. Appends one row to results.tsv."""
import csv, time, os, sys
from pathlib import Path
from ast_mapper import map_project
from gkg_navigator import GKGNavigator
from ollama_client import OllamaClient
from ab_quests import QUESTS
from ab_runner import run_quest_ab, score_metrics
from ab_stats import quality_table

REPO_PATH  = "_demo_repos/Sepia"
MODEL      = "qwen2.5-coder:1.5b"
TSV        = "results.tsv"

client = OllamaClient(MODEL)
g      = map_project(REPO_PATH)
nav    = GKGNavigator(g, REPO_PATH)

results, t0 = [], time.time()
for q in QUESTS:
    bf, bn, gn = run_quest_ab(q, client, REPO_PATH, nav)
    score_metrics(bf, q, nav)
    score_metrics(bn, q, nav)
    score_metrics(gn, q, nav)
    results.append((bf, bn, gn))

from ab_stats import _get_quality
import statistics

def _score(results):
    vals = [_get_quality(m) for triple in results for m in triple if _get_quality(m) >= 0]
    toks = [m.total_tokens   for triple in results for m in triple]
    mean_q = statistics.mean(vals) if vals else 0.0
    mean_t = statistics.mean(toks) if toks else 0
    return mean_q, mean_t, mean_q / max(mean_t, 1) * 1000

mean_q, mean_t, efficiency = _score(results)

# print table
print(quality_table(results, QUESTS))
print(f"\nSCORE={mean_q:.4f}  TOKENS={mean_t:.0f}  EFFICIENCY={efficiency:.4f}")

# append to tsv
row = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "score":      f"{mean_q:.4f}",
    "tokens":     f"{mean_t:.0f}",
    "efficiency": f"{efficiency:.6f}",
    "elapsed_s":  f"{time.time()-t0:.1f}",
    "note":       sys.argv[1] if len(sys.argv) > 1 else "baseline",
}
write_header = not Path(TSV).exists()
with open(TSV, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=row.keys(), delimiter="\t")
    if write_header:
        w.writeheader()
    w.writerow(row)

print(f"\nAppended to {TSV}")
