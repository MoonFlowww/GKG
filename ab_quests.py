"""A/B Quest definitions and runner scaffold.

Each quest runs in two conditions:
  BASELINE — AI gets raw file tree + reads files on demand
  GKG      — AI gets GKGNavigator interface

Metrics collected per run:
  turns          : number of LLM round-trips
  input_tokens   : total tokens sent to LLM
  output_tokens  : total tokens received from LLM
  files_opened   : files read (baseline) or code fetches (gkg)
  success        : bool — did the AI produce the correct output?
  notes          : freeform observation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Quest:
    id: int
    name: str
    prompt: str                     # what the AI is asked
    target_node: Optional[str]      # MESO/MICRO name relevant to the task (hint for verifier)
    success_criteria: str           # human-readable check
    complexity: str                 # retrieval | local_add | cross_module | new_feature | analysis
    gen_keywords: list = field(default_factory=list)  # expected identifiers for generation scoring


QUESTS: list[Quest] = [
    Quest(
        id=0,
        name="Find & Repeat Function",
        prompt=(
            "Find where this codebase computes tick spacing for a plot axis — "
            "the algorithm that takes a numeric range [lo, hi] and produces "
            "evenly-spaced tick values with nice round numbers. "
            "Return the complete source code of that function, unchanged."
        ),
        target_node="TickEngine.compute",
        success_criteria=(
            "Output contains the verbatim body of TickEngine::compute. "
            "No paraphrasing, no omissions."
        ),
        complexity="retrieval",
    ),
    Quest(
        id=1,
        name="Scientific Notation on Axis",
        prompt=(
            "Add a method `set_scientific_notation(bool enabled, int precision=2)` "
            "to the axis tick formatting logic. When enabled, labels should be formatted "
            "as e.g. '1.23e+04' instead of '12300'. Show the modified class."
        ),
        target_node="TickEngine",
        success_criteria=(
            "New method added to the correct class. "
            "Uses snprintf/std::format or equivalent for scientific notation. "
            "No unrelated changes."
        ),
        complexity="local_add",
        gen_keywords=["set_scientific_notation", "scientific_notation", "snprintf", "precision"],
    ),
    Quest(
        id=2,
        name="Line/Scatter Mode Switch",
        prompt=(
            "Make line plots switchable between line mode (current behaviour, points connected) "
            "and scatter mode (points only, no connecting lines, but preserving series identity "
            "so multi-series plots still show distinct point families). "
            "Add a `PlotMode` enum and a `set_mode(PlotMode)` method to the relevant class."
        ),
        target_node="Figure",
        success_criteria=(
            "PlotMode enum with at least LINE/SCATTER values. "
            "`set_mode` wired into the rendering path. "
            "Series identity preserved in scatter mode."
        ),
        complexity="cross_module",
        gen_keywords=["PlotMode", "set_mode", "SCATTER", "LINE"],
    ),
    Quest(
        id=3,
        name="Regression Option on Plot",
        prompt=(
            "Add a regression overlay option for line plots supporting three modes: "
            "LINEAR (least-squares line), SPLINE (cubic interpolation), and FRONTIER "
            "(convex-hull upper/lower envelope). "
            "Add `add_regression(RegressionMode mode)` to the figure API. "
            "Math can be self-contained (no external libs)."
        ),
        target_node="Figure",
        success_criteria=(
            "RegressionMode enum (LINEAR/SPLINE/FRONTIER). "
            "`add_regression` triggers a separate render pass. "
            "LINEAR uses closed-form OLS. SPLINE uses basic cubic. "
            "FRONTIER computes convex-hull upper/lower bounds."
        ),
        complexity="new_feature",
        gen_keywords=["RegressionMode", "add_regression", "LINEAR", "SPLINE", "FRONTIER"],
    ),
    Quest(
        id=4,
        name="Histogram Plot",
        prompt=(
            "Create a new plot type `Histogram` analogous to the existing line plot. "
            "It takes a data vector, a bin count, and optional range [min, max]. "
            "Bins are drawn as filled rectangles. "
            "Integrate it with the existing canvas/rendering pipeline."
        ),
        target_node="Canvas",
        success_criteria=(
            "New Histogram class (or struct) defined. "
            "Uses Canvas primitives (fill_rect or equivalent) for bar drawing. "
            "Bin computation is correct (uniform width, boundary-inclusive). "
            "Plugs into Figure or standalone render."
        ),
        complexity="new_feature",
        gen_keywords=["Histogram", "fill_rect", "bin", "bins"],
    ),
    Quest(
        id=5,
        name="Low-Latency Optimisation Audit",
        prompt=(
            "Analyse the codebase for low-latency improvement opportunities. "
            "Focus on: SIMD/auto-vectorisation hints, cache-line alignment, "
            "branch prediction attributes (__builtin_expect / [[likely]]), "
            "STL container alternatives (flat_map, small-buffer optimisation), "
            "and compiler intrinsics. "
            "Return a prioritised list of specific changes with the target location "
            "(class::method) and the expected gain for each."
        ),
        target_node=None,
        success_criteria=(
            "At least 5 concrete suggestions. "
            "Each references a real class::method from the codebase. "
            "Suggestions are actionable (specific attribute/intrinsic/container named). "
            "Priorities are justified (hot-path vs cold-path reasoning)."
        ),
        complexity="analysis",
        gen_keywords=["__builtin_expect", "likely", "simd", "alignment", "cache_line", "flat_map"],
    ),
]


# ── metrics ──────────────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    quest_id: int
    condition: str          # "baseline_full" | "baseline_nav" | "gkg_nav"
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    files_opened: int = 0   # raw file reads (baseline) or get_code calls (gkg)
    notes: str = ""
    latency_s: float = 0.0      # wall-clock seconds (from OllamaClient.stats_summary elapsed_s)
    # verification scores (filled by verify_answer / score_metrics)
    recall: float = -1.0
    precision: float = -1.0
    f1: float = -1.0
    # LLM judge scores (filled by score_with_judge)
    judge_score: float = -1.0
    judge_reason: str = ""
    # full turn-by-turn conversation log: list of {"role", "content", "tokens"} dicts
    conversation: list = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def quality(self) -> float:
        """Primary quality score: judge_score for gen/analysis, f1 for retrieval."""
        if self.judge_score >= 0:
            return self.judge_score
        if self.f1 >= 0:
            return self.f1
        return -1.0

    def summary(self) -> str:
        q_str = "{:.0%}".format(self.quality) if self.quality >= 0 else "?"
        return (
            "{:14s}  turns={:2d}  tok={:5d}  lat={:.1f}s  q={}".format(
                self.condition, self.turns, self.total_tokens, self.latency_s, q_str
            )
        )


@dataclass
class ABResult:
    quest: Quest
    baseline: RunMetrics
    gkg: RunMetrics

    def delta(self) -> str:
        dt = self.gkg.turns - self.baseline.turns
        dtok = self.gkg.total_tokens - self.baseline.total_tokens
        df = self.gkg.files_opened - self.baseline.files_opened
        sign = lambda x: ("+" if x > 0 else "") + str(x)
        return (
            f"Q{self.quest.id} [{self.quest.complexity:12s}] {self.quest.name}\n"
            f"  baseline : {self.baseline.summary()}\n"
            f"  gkg      : {self.gkg.summary()}\n"
            f"  delta    : turns={sign(dt)}  tokens={sign(dtok)}  files={sign(df)}"
        )


# ── runner scaffold ───────────────────────────────────────────────────────────

def run_ab(
    quest: Quest,
    baseline_fn,   # callable(quest) -> RunMetrics
    gkg_fn,        # callable(quest, navigator) -> RunMetrics
    navigator,     # GKGNavigator instance
) -> ABResult:
    """Execute one quest in both conditions, return combined result."""
    b = baseline_fn(quest)
    g = gkg_fn(quest, navigator)
    return ABResult(quest=quest, baseline=b, gkg=g)


def print_report(results: list[ABResult]) -> None:
    print("\n" + "=" * 80)
    print("A/B RESULTS")
    print("=" * 80)
    for r in results:
        print(r.delta())
        print()

    # aggregate
    b_tok = sum(r.baseline.total_tokens for r in results)
    g_tok = sum(r.gkg.total_tokens for r in results)
    b_turns = sum(r.baseline.turns for r in results)
    g_turns = sum(r.gkg.turns for r in results)
    print(f"TOTAL  baseline: {b_turns} turns / {b_tok} tokens")
    print(f"TOTAL  gkg     : {g_turns} turns / {g_tok} tokens")
    pct = (g_tok - b_tok) / b_tok * 100 if b_tok else 0
    print(f"TOKEN DELTA    : {pct:+.1f}%")
