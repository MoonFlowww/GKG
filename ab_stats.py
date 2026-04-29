"""Statistical utilities for A/B test reporting.

Intentionally dependency-free (stdlib only).
scipy/numpy are NOT required — bootstrap CI is used instead of parametric tests
because N=6 quests is too small for t-tests to be meaningful.
"""
from __future__ import annotations

import random
import statistics
import math
from typing import Sequence


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap mean and confidence interval.

    Returns (mean, lo, hi).
    Excludes sentinel values (< 0) before computing.
    """
    vals = [v for v in values if v >= 0]
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0])
    boots = sorted(
        statistics.mean(random.choices(vals, k=len(vals)))
        for _ in range(n_boot)
    )
    lo_i = int((1 - ci) / 2 * n_boot)
    hi_i = int((1 + ci) / 2 * n_boot)
    return (statistics.mean(vals), boots[lo_i], boots[hi_i])


def effect_size(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d between two samples (a=treatment, b=control). Excludes sentinels."""
    av = [v for v in a if v >= 0]
    bv = [v for v in b if v >= 0]
    if len(av) < 2 or len(bv) < 2:
        return float("nan")
    mean_diff = statistics.mean(av) - statistics.mean(bv)
    pooled_var = (statistics.variance(av) + statistics.variance(bv)) / 2
    return mean_diff / math.sqrt(pooled_var) if pooled_var > 0 else float("nan")


def significance_caveat(n_quests: int, n_runs: int = 1) -> str:
    total = n_quests * n_runs
    if total >= 30:
        return "[N={} — sufficient for parametric testing]".format(total)
    needed = max(0, 30 - total)
    return (
        "[N={} quest-runs — bootstrap CI only, results indicative not conclusive. "
        "Need ~{} more runs for parametric significance testing.]"
    ).format(total, needed)


def quality_table(
    all_results: list,  # list of (bf, bn, gn) RunMetrics triples
    quests: list,
) -> str:
    """Print a compact quality/token/latency summary table."""
    H = "{:28s}  {:12s}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}  {:>5}  {:>5}  {:>5}  {:>6}"
    SEP = "-" * 118
    header = H.format(
        "Quest", "Cplx",
        "F-qual", "N-qual", "G-qual",
        "F-tok", "N-tok", "G-tok",
        "F-lat", "N-lat", "G-lat",
        "Winner",
    )

    def _q(m) -> str:
        v = m.quality
        return "{:.0%}".format(v) if v >= 0 else "?"

    def _lat(m) -> str:
        return "{:.0f}s".format(m.latency_s) if m.latency_s > 0 else "?"

    def _winner(bf, bn, gn) -> str:
        scores = [("F", bf.quality), ("N", bn.quality), ("G", gn.quality)]
        valid = [(c, s) for c, s in scores if s >= 0]
        if not valid:
            return "?"
        best_score = max(s for _, s in valid)
        winners = [c for c, s in valid if abs(s - best_score) < 0.001]
        if len(winners) == 3:
            # tie on quality — pick lowest tokens
            toks = {"F": bf.total_tokens, "N": bn.total_tokens, "G": gn.total_tokens}
            return min(winners, key=lambda c: toks[c])
        return "/".join(winners)

    lines = [header, SEP]
    bf_quals, bn_quals, gn_quals = [], [], []
    bf_toks, bn_toks, gn_toks = [], [], []
    bf_lats, bn_lats, gn_lats = [], [], []

    for (bf, bn, gn), q in zip(all_results, quests):
        name = "Q{} {}".format(q.id, q.name)
        lines.append(H.format(
            name[:28], q.complexity[:12],
            _q(bf), _q(bn), _q(gn),
            bf.total_tokens, bn.total_tokens, gn.total_tokens,
            _lat(bf), _lat(bn), _lat(gn),
            _winner(bf, bn, gn),
        ))
        bf_quals.append(bf.quality); bn_quals.append(bn.quality); gn_quals.append(gn.quality)
        bf_toks.append(bf.total_tokens); bn_toks.append(bn.total_tokens); gn_toks.append(gn.total_tokens)
        bf_lats.append(bf.latency_s); bn_lats.append(bn.latency_s); gn_lats.append(gn.latency_s)

    lines.append(SEP)

    # summary stats
    def _ci_str(vals):
        m, lo, hi = bootstrap_ci(vals)
        if math.isnan(m): return "N/A"
        return "{:.0%} [{:.0%}-{:.0%}]".format(m, lo, hi)

    def _mean_tok(vals):
        v = [x for x in vals if x > 0]
        return "{:.0f}".format(statistics.mean(v)) if v else "?"

    def _mean_lat(vals):
        v = [x for x in vals if x > 0]
        return "{:.1f}s".format(statistics.mean(v)) if v else "?"

    def _eff(a_toks, b_toks):
        at = [x for x in a_toks if x > 0]
        bt = [x for x in b_toks if x > 0]
        if not at or not bt: return "?"
        return "{:.1f}x".format(statistics.mean(bt) / statistics.mean(at))

    lines.append("")
    lines.append("Quality 95% CI (bootstrap):")
    lines.append("  full : {}".format(_ci_str(bf_quals)))
    lines.append("  nav  : {}".format(_ci_str(bn_quals)))
    lines.append("  gkg  : {}".format(_ci_str(gn_quals)))
    lines.append("")
    lines.append("Mean tokens :  full={}  nav={}  gkg={}".format(
        _mean_tok(bf_toks), _mean_tok(bn_toks), _mean_tok(gn_toks)))
    lines.append("Mean latency:  full={}  nav={}  gkg={}".format(
        _mean_lat(bf_lats), _mean_lat(bn_lats), _mean_lat(gn_lats)))
    lines.append("Token savings: gkg vs full={}  gkg vs nav={}".format(
        _eff(gn_toks, bf_toks), _eff(gn_toks, bn_toks)))
    lines.append("")

    # effect sizes
    d_gkg_nav  = effect_size(gn_quals, bn_quals)
    d_gkg_full = effect_size(gn_quals, bf_quals)
    if not math.isnan(d_gkg_nav):
        lines.append("Cohen's d (quality):  gkg vs nav={:.2f}  gkg vs full={:.2f}".format(
            d_gkg_nav, d_gkg_full))
        lines.append("")

    lines.append(significance_caveat(len(all_results)))
    return "\n".join(lines)
