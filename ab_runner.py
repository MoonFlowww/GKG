"""A/B agent runners.

Three conditions:
  baseline_full  — LLM gets entire codebase upfront, answers in one shot
  baseline_nav   — LLM gets file tree, navigates via LIST/READ commands
  gkg_nav        — LLM gets GKG MACRO map + auto-route hint, navigates via
                   CD/EDGES/GET_CODE/FIND commands

Multi-turn: proper role/content message array — each turn appends
  {"role":"user","content": tool_result}
  {"role":"assistant","content": model_response}
so the model sees a real dialogue, not a growing single-message wall.
"""
from __future__ import annotations

import ast
import math
import os
import pickle
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ab_quests import GKGSideMetrics, Quest, RunMetrics
from gkg_navigator import GKGNavigator
from ollama_client import OllamaClient

MAX_TURNS        = 20
MAX_FETCHES      = 5       # hard cap on READ/GET_CODE before forcing ANSWER
FILE_CAP         = 50000   # chars per file read (baseline_nav)
CODE_CAP         = 30000   # chars per GET_CODE result
MAX_CTX_CHARS    = 14000  # trim history above this to prevent Q2-style token explosion

# ── verification ──────────────────────────────────────────────────────────────

import re as _re

def _tokenize(text: str) -> set[str]:
    """Extract meaningful alphanumeric tokens (len>=2, skip pure numbers)."""
    return {t for t in _re.findall(r'[A-Za-z_][A-Za-z0-9_]*', text) if len(t) >= 2}


def detect_language(code: str) -> str:
    """Detect primary language from code-fence tags, then structural heuristics."""
    m = re.search(r'```(\w+)', code)
    if m:
        tag = m.group(1).lower()
        if tag in ("python", "py"):
            return "python"
        if tag in ("cpp", "c++", "cc", "cxx", "c"):
            return "cpp"
    if "def " in code and ("self" in code or "import " in code):
        return "python"
    if "#include" in code or "::" in code:
        return "cpp"
    return "cpp"   # default: this project is C++


def _extract_code_blocks(text: str) -> str:
    """Return code-fence contents joined, or full text if no fences."""
    blocks = re.findall(r'```[\w]*\n?(.*?)```', text, re.DOTALL)
    return "\n\n".join(blocks) if blocks else text


def check_working_rate(notes: str, language: str) -> dict:
    """Syntax/compile gate. Returns {"passes": bool, "stage": str, "error": str}.

    Operates on code extracted from markdown fences.
    Currently implements: syntax (Python: ast.parse; C++: g++ -fsyntax-only).
    """
    if not notes or notes.startswith("["):
        return {"passes": False, "stage": "empty", "error": "no code produced"}

    code = _extract_code_blocks(notes)
    if not code.strip():
        return {"passes": False, "stage": "empty", "error": "no code produced"}

    if language == "python":
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {"passes": False, "stage": "syntax", "error": str(e)[:200]}
        return {"passes": True, "stage": "syntax", "error": ""}

    if language in ("cpp", "c"):
        if not shutil.which("g++"):
            return {"passes": True, "stage": "skip", "error": "g++ not found"}
        body = code if "int main" in code else "#include <cstdio>\n" + code + "\nint main(){}"
        with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False, mode="w") as f:
            f.write(body)
            fname = f.name
        try:
            r = subprocess.run(
                ["g++", "-std=c++17", "-fsyntax-only", "-w", fname],
                capture_output=True, timeout=10,
            )
        finally:
            os.unlink(fname)
        passes = r.returncode == 0
        return {
            "passes": passes,
            "stage": "syntax",
            "error": r.stderr.decode()[:300] if not passes else "",
        }

    return {"passes": True, "stage": "skip", "error": "unsupported language"}


def _trim_messages(messages: list[dict]) -> list[dict]:
    """Keep seed + code-bearing tool results + last 4 messages when context grows too large."""
    total = sum(len(m.get("content", "")) for m in messages)
    if total <= MAX_CTX_CHARS or len(messages) <= 5:
        return messages
    # preserve tool results that contain actual source code (heuristic: long + has braces)
    code_msgs = [m for m in messages[1:-4]
                 if m.get("role") == "tool"
                 and len(m.get("content", "")) > 200
                 and "{" in m.get("content", "")]
    marker = {"role": "user", "content": "[...earlier navigation trimmed...]"}
    return [messages[0]] + code_msgs[:2] + [marker] + messages[-4:]


def get_ground_truth(quest, navigator: "GKGNavigator") -> str:
    """Return the real source code for a quest's target_node via GET_CODE."""
    if quest.target_node is None:
        return ""
    return navigator.get_code(quest.target_node)


def verify_answer(answer: str, ground_truth: str) -> dict:
    """Token-overlap verification (non-exclusive: tokens can appear anywhere).

    Returns:
      recall    — fraction of ground-truth tokens found in answer
      precision — fraction of answer tokens that are in ground-truth
      f1        — harmonic mean
      found     — set of matched tokens (for inspection)
      missing   — set of ground-truth tokens absent from answer
    """
    if not ground_truth or not answer:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0,
                "found": set(), "missing": set()}

    gt_tokens  = _tokenize(ground_truth)
    ans_tokens = _tokenize(answer)

    found   = gt_tokens & ans_tokens
    missing = gt_tokens - ans_tokens

    recall    = len(found) / len(gt_tokens)  if gt_tokens  else 0.0
    precision = len(found) / len(ans_tokens) if ans_tokens else 0.0
    f1 = (2 * recall * precision / (recall + precision)
          if (recall + precision) > 0 else 0.0)

    return {
        "recall":    round(recall,    3),
        "precision": round(precision, 3),
        "f1":        round(f1,        3),
        "found":     found,
        "missing":   missing,
    }

# ── system prompts ────────────────────────────────────────────────────────────

_BASELINE_FULL_SYS = """\
You are a code assistant. The full codebase is provided above.
Rules:
- Read the codebase carefully before answering.
- Fulfill the task exactly as asked:
    - If asked to FIND/RETURN code: paste the verbatim source. Never paraphrase.
    - If asked to ADD/CREATE/MODIFY: generate the new or modified code based on existing patterns.
    - If asked to ANALYSE: provide a detailed analysis with specific file/class references.
- Format code in ```cpp (or appropriate language) blocks.
- If a target does not exist in the codebase, say "NOT FOUND".
"""

_BASELINE_NAV_SYS = """\
You are a code exploration agent. A full list of source files is provided above.
Issue ONE command per turn. Available commands:

  LIST: <rel_path>    list contents of a directory
  READ: <rel_path>    read a source file
  ANSWER: <text>      final answer — use when you have enough information

Rules:
- Output ONLY the command line, nothing else before or after.
- The full file list is already given — read it to pick the right file directly.
- Read the relevant source file(s) before answering.
- ANSWER must directly fulfill the task:
    - If asked to FIND/RETURN code: paste the verbatim source.
    - If asked to ADD/CREATE/MODIFY: generate the new or modified code.
- Use at most 5 navigation commands, then ANSWER.
- ANSWER format: write ANSWER: on one line, then paste the code on the following lines.
"""

_GKG_NAV_SYS = """\
You are a code exploration agent using a GKG Knowledge-Graph Navigator.
The GKG is your map — use it to navigate to the right code, then answer.

Issue ONE command per turn. Available commands:

  CD: <name>            enter a module or class — shows its full contents
  UP                    go one level up
  RELATIONS: <name>     show all incoming and outgoing edges for any named node
  LIST_FILE: <path>     list all classes and methods in a file with line ranges
  GET_CODE: <name>      get source of a class or method  e.g. TickEngine.compute
  FIND: <keyword>       search nodes by name or intent
  ANSWER: <text>        final answer — use when you have enough information

Output ONLY the command line, nothing else.
Use CD to see module structure. Use LIST_FILE to see classes/methods with line numbers.
Use GET_CODE to read actual source before answering.
ANSWER must directly fulfill the task:
  - If asked to FIND/RETURN code: paste the verbatim source.
  - If asked to ADD/CREATE/MODIFY: generate the new or modified code.
  - If asked to ANALYSE: provide the analysis.
Use at most 5 navigation commands, then ANSWER.
ANSWER format: write ANSWER: on one line, then paste the code or analysis on the following lines.
"""

# ── helpers ───────────────────────────────────────────────────────────────────

def _file_tree(project_path: str, rel: str = ".") -> str:
    """Single-level directory listing (used by LIST: command)."""
    root = Path(project_path).resolve()
    target = (root / rel).resolve()
    lines: list[str] = []
    try:
        for entry in sorted(target.iterdir()):
            rel_entry = entry.relative_to(root)
            lines.append("  " + str(rel_entry).replace("\\", "/") + ("/" if entry.is_dir() else ""))
    except Exception as e:
        return "[error listing {}: {}]".format(rel, e)
    return "\n".join(lines) if lines else "(empty)"


def _file_tree_full(project_path: str) -> str:
    """Recursive flat listing of all source files — shows full path structure at a glance."""
    root = Path(project_path).resolve()
    IGNORE = {".git", "__pycache__", "build", "dist", "node_modules", ".venv", "venv"}
    EXTS = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp",
            ".py", ".js", ".ts", ".tsx", ".go", ".rs",
            ".java", ".cs", ".rb", ".swift", ".kt"}
    lines: list[str] = []
    for fpath in sorted(root.rglob("*")):
        if not fpath.is_file():
            continue
        if any(p in IGNORE for p in fpath.parts):
            continue
        if fpath.suffix.lower() not in EXTS:
            continue
        lines.append("  " + str(fpath.relative_to(root)).replace("\\", "/"))
    return "\n".join(lines) if lines else "(no source files found)"


def _read_file(project_path: str, rel: str, cap: int = FILE_CAP) -> str:
    p = Path(project_path).resolve() / rel
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > cap:
            content = content[:cap] + "\n... [truncated — {} chars total]".format(len(content))
        return content
    except Exception as e:
        return "[error reading {}: {}]".format(rel, e)


def _all_sources(project_path: str) -> str:
    """Concat all header/source files for baseline_full.

    Priority order: headers first (most likely to contain definitions),
    then implementation files, then examples last.
    Per-file cap: 3k chars. No total cap — model gets everything.
    """
    EXTS = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".py"}
    IGNORE = {".git", "__pycache__", "build", "dist", "node_modules", "stress"}
    FILE_CAP = 3000

    root = Path(project_path).resolve()
    all_files: list[Path] = []
    for fpath in sorted(root.rglob("*")):
        if not fpath.is_file():
            continue
        if any(p in IGNORE for p in fpath.parts):
            continue
        if fpath.suffix.lower() not in EXTS:
            continue
        all_files.append(fpath)

    # headers first, then .cpp, then examples
    def _priority(p: Path) -> int:
        parts_lower = [x.lower() for x in p.parts]
        if "examples" in parts_lower or "demo" in parts_lower:
            return 2
        if p.suffix.lower() in (".hpp", ".h"):
            return 0
        return 1

    all_files.sort(key=lambda p: (_priority(p), str(p)))

    parts: list[str] = []
    for fpath in all_files:
        rel = str(fpath.relative_to(root)).replace("\\", "/")
        try:
            src = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(src) > FILE_CAP:
            src = src[:FILE_CAP] + "\n// ... [file truncated]"
        parts.append("// === {} ===\n{}".format(rel, src))

    return "\n\n".join(parts)


def _auto_route(navigator: GKGNavigator, task: str) -> str:
    """Extract keywords from task, find relevant GKG nodes, return as hint."""
    # pull likely identifiers (CamelCase, snake_case words 4+ chars)
    words = re.findall(r'`([^`]+)`|([A-Z][a-z]+[A-Za-z]*)|([a-z_]{4,}\w*)', task)
    keywords: list[str] = []
    for groups in words:
        for g in groups:
            if g and g not in keywords:
                keywords.append(g)

    hits: list[str] = []
    seen: set[str] = set()
    for kw in keywords[:6]:
        result = navigator.find(kw)
        if "No nodes" not in result:
            for line in result.splitlines()[1:6]:   # skip header line
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    hits.append(line)

    if not hits:
        return ""
    return "[AUTO-ROUTE: GKG found these nodes relevant to the task]\n" + "\n".join(hits[:10])


# ── command parser ────────────────────────────────────────────────────────────

_CMD_RE = re.compile(
    r'^\s*(LIST|READ|CD|EDGES|RELATIONS|LIST_FILE|GET_CODE|FIND|ANSWER|UP)\s*:?\s*(.*)',
    re.IGNORECASE | re.DOTALL,
)


def _parse(response: str) -> tuple[str, str]:
    lines = response.strip().splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper() == "UP":
            return "UP", ""
        m = _CMD_RE.match(line.strip())
        if m:
            cmd = m.group(1).upper()
            arg = m.group(2).strip()
            if cmd == "ANSWER" and not arg:
                # model put code on subsequent lines after ANSWER:
                rest = "\n".join(lines[i + 1:]).strip()
                return cmd, rest
            # strip trailing explanations for lookup commands
            if cmd not in ("ANSWER", "FIND"):
                arg = arg.split()[0] if arg.split() else arg
            return cmd, arg
    return "UNKNOWN", response.strip()


# ── helpers ───────────────────────────────────────────────────────────────────

def _f1_prog(v: float) -> str:
    if v < -2.5: return "N/A"
    if v < -1.5: return "gen"
    if v < 0:    return "?"
    return "{:.0%}".format(v)


def _task_type_tag(complexity: str) -> str:
    if complexity in ("local_add", "cross_module", "new_feature"):
        return "\n\n[TASK TYPE: GENERATION — your ANSWER must contain new or modified code, not existing code verbatim]"
    if complexity == "analysis":
        return "\n\n[TASK TYPE: ANALYSIS — your ANSWER must contain a prioritised list with specific class::method references]"
    return ""


_GEN_NUDGE = (
    "\n\n[You now have the source context. "
    "For this task you must GENERATE new/modified code — "
    "do NOT return existing code unchanged. "
    "Write your implementation now as ANSWER:]"
)


# ── condition 1: baseline_full ────────────────────────────────────────────────

def run_baseline_full(quest: Quest, client: OllamaClient,
                      project_path: str) -> RunMetrics:
    """Single-shot: give all source upfront, one LLM call."""
    client.reset_stats()
    metrics = RunMetrics(quest_id=quest.id, condition="baseline_full")

    sources = _all_sources(project_path)
    prompt = (
        "CODEBASE:\n\n"
        + sources
        + "\n\n---\nTASK: "
        + quest.prompt
        + _task_type_tag(quest.complexity)
    )

    response = client.complete(prompt, system=_BASELINE_FULL_SYS,
                               max_tokens=2048, label="bf_q{}".format(quest.id))
    metrics.turns = 1
    metrics.files_opened = 0   # no navigation — all given
    metrics.notes = response.strip()[:8000]

    s = client.stats_summary()
    metrics.input_tokens  = s["prompt_tokens"]
    metrics.output_tokens = s["completion_tokens"]
    metrics.latency_s     = s["elapsed_s"]
    return metrics


# ── condition 2: baseline_nav ─────────────────────────────────────────────────

def run_baseline_nav(quest: Quest, client: OllamaClient,
                     project_path: str) -> RunMetrics:
    """Multi-turn: LLM navigates file tree via LIST/READ commands."""
    client.reset_stats()
    metrics = RunMetrics(quest_id=quest.id, condition="baseline_nav")

    tree = _file_tree_full(project_path)
    messages: list[dict] = [
        {"role": "user",
         "content": "TASK: {}{}\n\nSOURCE FILES:\n{}".format(
             quest.prompt, _task_type_tag(quest.complexity), tree)},
    ]
    reads_done: set[str] = set()
    last_cmd = ("", "")

    for _ in range(MAX_TURNS):
        response = client.chat(messages, system=_BASELINE_NAV_SYS,
                               max_tokens=256, label="bn_q{}".format(quest.id))
        metrics.turns += 1
        messages.append({"role": "assistant", "content": response})

        cmd, arg = _parse(response)

        if cmd == "ANSWER":
            metrics.notes = arg[:4000]
            metrics.conversation.append({"role": "assistant", "content": response, "cmd": "ANSWER"})
            break

        elif cmd == "LIST":
            target = Path(project_path).resolve() / (arg or ".")
            if not target.exists():
                result = "[not found: {}] — use LIST: . to see available directories.".format(arg)
            elif target.is_file():
                result = "[{} is a file — use READ: {} to read it]".format(arg, arg)
            else:
                result = "[DIR: {}]\n{}".format(arg, _file_tree(project_path, arg or "."))

        elif cmd == "READ":
            if arg in reads_done:
                result = "[already read {}] — use ANSWER if you have enough info.".format(arg)
            else:
                p = Path(project_path).resolve() / arg
                if not p.exists():
                    result = "[file not found: {}] — use LIST: . or LIST: include to find the correct path.".format(arg)
                else:
                    reads_done.add(arg)
                    metrics.files_opened += 1
                    content = _read_file(project_path, arg)
                    nudge = _GEN_NUDGE if quest.complexity != "retrieval" else "\n\n[File read. Use ANSWER: and paste the relevant code verbatim.]"
                    result = "[FILE: {}]\n{}{}".format(arg, content, nudge)
                    if metrics.files_opened >= MAX_FETCHES:
                        result += "\n[READ LIMIT REACHED ({}/{}). You MUST use ANSWER on your next turn.]".format(
                            metrics.files_opened, MAX_FETCHES)

        else:
            result = "[unknown command: {}  — use LIST, READ, or ANSWER]".format(cmd)

        # repeat guard
        if (cmd, arg) == last_cmd:
            result += "\n[You already issued this command. Use ANSWER now.]"
        last_cmd = (cmd, arg)

        metrics.conversation.append({"role": "assistant", "content": response, "cmd": "{}: {}".format(cmd, arg)})
        metrics.conversation.append({"role": "tool", "content": result})

        messages.append({"role": "user", "content": result})
        messages = _trim_messages(messages)

    else:
        metrics.notes = "[max turns reached]"

    s = client.stats_summary()
    metrics.input_tokens  = s["prompt_tokens"]
    metrics.output_tokens = s["completion_tokens"]
    metrics.latency_s     = s["elapsed_s"]
    return metrics


# ── condition 3: gkg_nav ──────────────────────────────────────────────────────

def run_gkg_nav(quest: Quest, client: OllamaClient,
                navigator: GKGNavigator) -> RunMetrics:
    """Multi-turn: LLM navigates via GKG commands. Auto-route seeds the context."""
    client.reset_stats()
    navigator._current = None
    metrics = RunMetrics(quest_id=quest.id, condition="gkg_nav")

    macro_map = navigator.dump()
    auto_hint = _auto_route(navigator, quest.prompt)

    seed = "TASK: {}{}\n\n[GKG MAP — MACRO LEVEL]\n{}".format(
        quest.prompt, _task_type_tag(quest.complexity), macro_map)
    if auto_hint:
        seed += "\n\n" + auto_hint

    messages: list[dict] = [{"role": "user", "content": seed}]
    last_cmd = ("", "")

    for _ in range(MAX_TURNS):
        response = client.chat(messages, system=_GKG_NAV_SYS,
                               max_tokens=256, label="gn_q{}".format(quest.id))
        metrics.turns += 1
        messages.append({"role": "assistant", "content": response})

        cmd, arg = _parse(response)

        if cmd == "ANSWER":
            metrics.notes = arg[:4000]
            metrics.conversation.append({"role": "assistant", "content": response, "cmd": "ANSWER"})
            break

        elif cmd == "CD":
            result = navigator.cd(arg)

        elif cmd == "UP":
            result = navigator.up()

        elif cmd == "EDGES":
            result = navigator.edges(arg if arg else None)

        elif cmd == "RELATIONS":
            result = navigator.relations(arg)

        elif cmd == "LIST_FILE":
            result = navigator.list_file(arg)

        elif cmd == "GET_CODE":
            raw = navigator.get_code(arg)
            metrics.files_opened += 1
            if len(raw) > CODE_CAP:
                raw = raw[:CODE_CAP] + "\n... [truncated]"
            nudge = _GEN_NUDGE if quest.complexity != "retrieval" else "\n\n[Source retrieved. Use ANSWER to respond.]"
            result = raw + nudge
            if metrics.files_opened >= MAX_FETCHES:
                result += "\n[GET_CODE LIMIT REACHED ({}/{}). You MUST use ANSWER on your next turn.]".format(
                    metrics.files_opened, MAX_FETCHES)

        elif cmd == "FIND":
            result = navigator.find(arg)

        else:
            result = "[unknown: {}  — use CD, UP, EDGES, GET_CODE, FIND, or ANSWER]".format(cmd)

        # repeat guard
        if (cmd, arg) == last_cmd:
            result += "\n[Repeated command detected. You have the info — use ANSWER now.]"
        last_cmd = (cmd, arg)

        metrics.conversation.append({"role": "assistant", "content": response, "cmd": "{}: {}".format(cmd, arg)})
        metrics.conversation.append({"role": "tool", "content": result})

        messages.append({"role": "user", "content": result})
        messages = _trim_messages(messages)

    else:
        metrics.notes = "[max turns reached]"

    s = client.stats_summary()
    metrics.input_tokens  = s["prompt_tokens"]
    metrics.output_tokens = s["completion_tokens"]
    metrics.latency_s     = s["elapsed_s"]
    return metrics


# ── convenience runners ───────────────────────────────────────────────────────

# aliases kept for notebook back-compat
def run_baseline(quest, client, project_path):
    return run_baseline_nav(quest, client, project_path)

def run_gkg(quest, client, navigator):
    return run_gkg_nav(quest, client, navigator)


def _kw_score(notes: str, gen_kw: list) -> float:
    """Fraction of gen_keywords found in code blocks (fallback: full text)."""
    import re as _r
    blocks = _r.findall(r'```[\w]*\n?(.*?)```', notes, _r.DOTALL)
    search = '\n'.join(blocks).lower() if blocks else notes.lower()
    hits = sum(1 for kw in gen_kw if kw.lower() in search)
    return round(hits / len(gen_kw), 3)


_CODE_COMPLEXITIES = {"retrieval", "local_add", "cross_module", "new_feature"}


def score_metrics(metrics: "RunMetrics", quest, navigator: "GKGNavigator") -> None:
    """Fill recall/precision/f1 in-place, gated by working_rate for code tasks."""
    gen_kw = getattr(quest, "gen_keywords", [])

    # working-rate gate: skip for analysis (prose output, not compilable code)
    if quest.complexity in _CODE_COMPLEXITIES:
        lang = detect_language(metrics.notes)
        wr = check_working_rate(metrics.notes, lang)
        metrics.working_rate = 1.0 if wr["passes"] else 0.0
        metrics.working_stage = wr["stage"]
        if not wr["passes"]:
            metrics.recall = metrics.precision = metrics.f1 = 0.0
            return

    if quest.complexity == "retrieval":
        if quest.target_node is None:
            metrics.recall = metrics.precision = metrics.f1 = -3.0
            return
        gt = get_ground_truth(quest, navigator)
        if not gt:
            metrics.recall = metrics.precision = metrics.f1 = -3.0
            return
        gt_body = gt.split("\n", 1)[1] if "\n" in gt else gt
        v = verify_answer(metrics.notes, gt_body)
        metrics.recall    = v["recall"]
        metrics.precision = v["precision"]
        metrics.f1        = v["f1"]
    else:
        # generation / analysis: keyword presence inside code fences
        if gen_kw and metrics.notes:
            score = _kw_score(metrics.notes, gen_kw)
            metrics.recall = metrics.precision = metrics.f1 = score
        else:
            metrics.recall = metrics.precision = metrics.f1 = (
                -2.0 if quest.target_node is not None else -3.0
            )


def score_structural(generated_code: str, quest, navigator) -> dict:
    """Compare generated code's GKG against expected blueprint nodes."""
    import tempfile, os
    if not generated_code or quest.target_node is None:
        return {"node_coverage": 0.0, "edge_faithfulness": 0.0}

    with tempfile.TemporaryDirectory() as tmp:
        fpath = os.path.join(tmp, "generated.py")
        with open(fpath, "w") as f:
            f.write(generated_code)
        try:
            from ast_mapper import map_project
            gen_graph = map_project(tmp)
        except Exception:
            return {"node_coverage": 0.0, "edge_faithfulness": 0.0}

    gen_names = {n.name.split("::")[-1].split(".")[-1].lower()
                 for n in gen_graph.nodes.values()}
    expected = navigator.find(quest.target_node)
    expected_names = {
        line.strip().split("—")[0].strip().lower()
        for line in expected.splitlines()[1:]
        if line.strip()
    }
    if not expected_names:
        return {"node_coverage": 0.0, "edge_faithfulness": 0.0}

    coverage = len(gen_names & expected_names) / len(expected_names)
    return {"node_coverage": round(coverage, 3), "edge_faithfulness": -1.0}


def compute_gkg_side_metrics(metrics: "RunMetrics") -> GKGSideMetrics:
    """Parse conversation log to compute GKG navigation diagnostics.

    Conversation entries alternate: assistant (with "cmd" key) then tool (role="tool").
    The final ANSWER turn has no following tool entry.
    """
    sm = GKGSideMetrics()
    conv = metrics.conversation
    if not conv:
        return sm

    # Build (cmd_str, tool_result) pairs from the interleaved log
    nav_entries: list[tuple[str, str]] = []
    i = 0
    while i < len(conv):
        entry = conv[i]
        if entry.get("role") == "assistant" and "cmd" in entry:
            cmd = entry["cmd"]
            if i + 1 < len(conv) and conv[i + 1].get("role") == "tool":
                nav_entries.append((cmd, conv[i + 1].get("content", "")))
                i += 2
            else:
                nav_entries.append((cmd, ""))   # ANSWER or orphaned turn
                i += 1
        else:
            i += 1

    sm.nav_turns = len(nav_entries)
    if not nav_entries:
        return sm

    get_code_count = 0
    unknown_count = 0
    repeat_count = 0
    cd_entries: list[tuple[str, str]] = []

    for j, (cmd, result) in enumerate(nav_entries):
        cu = cmd.upper()
        if cu.startswith("GET_CODE"):
            get_code_count += 1
        if cu.startswith("UNKNOWN"):
            unknown_count += 1
        if cu.startswith("CD"):
            cd_entries.append((cmd, result))
        if j > 0 and cmd == nav_entries[j - 1][0]:
            repeat_count += 1

    _error_markers = ("error", "not found", "unknown command", "[unknown")
    cd_success = sum(
        1 for _, res in cd_entries
        if res and not any(m in res.lower() for m in _error_markers)
    )

    n = sm.nav_turns
    sm.get_code_count = get_code_count
    sm.gkg_call_rate = round(get_code_count / n, 3)
    sm.dead_end_rate = round(unknown_count / n, 3)
    sm.repeat_cmd_rate = round(repeat_count / max(n - 1, 1), 3) if n > 1 else 0.0
    sm.cd_success_rate = round(cd_success / len(cd_entries), 3) if cd_entries else 0.0
    return sm


# ── LLM judge ────────────────────────────────────────────────────────────────

JUDGE_MODEL = "gemma4:26b"

_JUDGE_SYS = """\
You are a strict code review judge. Evaluate whether the given answer fulfills the task.
Reply with ONLY a single JSON object on one line — no prose before or after.
Format: {"score": 0.0, "reason": "one sentence"}
score: 1.0=fully correct and complete, 0.5=partially correct or incomplete, 0.0=wrong/missing.
"""

def judge_answer(quest, answer: str, judge_client: "OllamaClient") -> dict:
    """Score answer using a thinking LLM judge. Returns {"score": float, "reason": str}."""
    import json as _json
    import re as _re
    if not answer or answer.startswith("["):
        return {"score": 0.0, "reason": "no answer produced"}

    # Trim to code blocks only — judge sees the diff, not noise
    blocks = _re.findall(r'```[\w]*\n?(.*?)```', answer, _re.DOTALL)
    code_excerpt = "\n".join(blocks)[:2000] if blocks else answer[:1500]

    prompt = (
        "TASK: {}\n\nCRITERIA: {}\n\nCODE:\n{}\n\n"
        "Reply ONLY with: {{\"score\": 0.0|0.5|1.0, \"reason\": \"one sentence\"}}"
    ).format(quest.prompt[:300], quest.success_criteria[:200], code_excerpt)

    try:
        raw = judge_client.complete(prompt, system=_JUDGE_SYS,
                                    max_tokens=80, label="judge_q{}".format(quest.id))
        m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if m:
            return _json.loads(m.group())
    except Exception as e:
        return {"score": -1.0, "reason": str(e)}
    return {"score": -1.0, "reason": "parse_error:" + raw[:40]}


_JUDGE_SYS_V2 = """\
You are a strict code reviewer. Score ONLY against the rubric below. Output JSON only — no prose.
{"correctness": 0|1|2, "completeness": 0|1|2, "no_hallucination": 0|1|2, "reason": "one sentence"}

Rubric:
  correctness:      2=fully correct logic, 1=mostly correct with minor errors, 0=wrong or broken
  completeness:     2=all required elements present, 1=partial, 0=missing key parts
  no_hallucination: 2=only references real constructs, 1=minor invented names, 0=fabricated API
"""


def judge_answer_v2(quest, answer: str, judge_client: "OllamaClient",
                    swap_check: bool = True) -> dict:
    """Three-dimension rubric judge with stability check.

    Runs the judge twice; if normalised scores differ by more than 1/6 (one rubric
    point), the result is flagged unstable and the average is returned.
    Returns {"score": float 0–1, "stable": bool, "reason": str}.
    """
    import json as _json
    if not answer or answer.startswith("["):
        return {"score": 0.0, "stable": True, "reason": "no answer produced"}

    blocks = re.findall(r'```[\w]*\n?(.*?)```', answer, re.DOTALL)
    excerpt = "\n".join(blocks)[:2000] if blocks else answer[:1500]

    prompt = "TASK: {}\n\nCRITERIA: {}\n\nCODE:\n{}".format(
        quest.prompt[:300], quest.success_criteria[:200], excerpt
    )

    def _call(p: str) -> tuple[dict, float]:
        try:
            raw = judge_client.complete(p, system=_JUDGE_SYS_V2,
                                        max_tokens=120, label="judge2_q{}".format(quest.id))
            m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            if m:
                d = _json.loads(m.group())
                total = (d.get("correctness", 0)
                         + d.get("completeness", 0)
                         + d.get("no_hallucination", 0))
                return d, round(total / 6.0, 4)
        except Exception as e:
            return {"reason": str(e)}, -1.0
        return {}, -1.0

    d_a, score_a = _call(prompt)
    if score_a < 0:
        return {"score": -1.0, "stable": False, "reason": "parse_error"}

    if swap_check:
        _, score_b = _call(prompt + "\n[Re-evaluate independently]")
        if score_b >= 0 and abs(score_a - score_b) > (1 / 6.0 + 0.001):
            avg = round((score_a + score_b) / 2.0, 4)
            return {"score": avg, "stable": False,
                    "reason": (d_a.get("reason", "") or "") + " [unstable]"}

    return {"score": score_a, "stable": True, "reason": d_a.get("reason", "")}


def score_with_judge(metrics: "RunMetrics", quest, judge_client=None) -> None:
    """Judge scoring with structured fallback when judge unavailable or fails.

    Uses judge_answer_v2 (3-dimension rubric + stability check) when a client
    is available; falls back to deterministic heuristics otherwise.
    """
    if judge_client is not None:
        result = judge_answer_v2(quest, metrics.notes, judge_client)
        score = result.get("score", -1.0)
        if score >= 0:
            metrics.judge_score = score
            metrics.judge_reason = result.get("reason", "")
            metrics.judge_stable = result.get("stable", True)
            return

    # Fallback: synthesize from existing deterministic scores
    gen_kw = getattr(quest, "gen_keywords", [])
    if quest.complexity == "retrieval":
        metrics.judge_score = metrics.f1 if metrics.f1 >= 0 else 0.0
        metrics.judge_reason = "fallback:f1"
    elif gen_kw and metrics.notes:
        kw = _kw_score(metrics.notes, gen_kw)
        f1 = max(metrics.f1, 0.0)
        metrics.judge_score = round(0.4 * f1 + 0.6 * kw, 3)
        metrics.judge_reason = f"fallback:kw={kw:.2f},f1={f1:.2f}"
    else:
        metrics.judge_score = 0.0
        metrics.judge_reason = "fallback:no-signal"


def _qual_str(m: "RunMetrics") -> str:
    q = m.quality
    if q < 0: return "?"
    return "{:.0%}".format(q)


def run_quest_ab(quest, client, project_path, navigator, judge_client=None):
    """Run all three conditions, score answers, print live progress.

    judge_client: optional OllamaClient for gemma4:26b judge scoring.
                  Runs after all three conditions to avoid polluting token stats.
    """
    print("  [full]  Q{}: {} ...".format(quest.id, quest.name))
    bf = run_baseline_full(quest, client, project_path)
    score_metrics(bf, quest, navigator)
    print("    turns={} tok={} lat={:.1f}s f1={}".format(
        bf.turns, bf.total_tokens, bf.latency_s, _f1_prog(bf.f1)))

    print("  [nav]   Q{}: {} ...".format(quest.id, quest.name))
    bn = run_baseline_nav(quest, client, project_path)
    score_metrics(bn, quest, navigator)
    print("    turns={} tok={} lat={:.1f}s f1={}".format(
        bn.turns, bn.total_tokens, bn.latency_s, _f1_prog(bn.f1)))

    print("  [gkg]   Q{}: {} ...".format(quest.id, quest.name))
    gn = run_gkg_nav(quest, client, navigator)
    score_metrics(gn, quest, navigator)
    gn.gkg_side = compute_gkg_side_metrics(gn)
    print("    turns={} tok={} lat={:.1f}s f1={} wr={}".format(
        gn.turns, gn.total_tokens, gn.latency_s, _f1_prog(gn.f1),
        "OK" if gn.working_rate >= 1.0 else "FAIL" if gn.working_rate == 0.0 else "?"))

    if judge_client is not None:
        print("  [judge] Q{} ...".format(quest.id))
        score_with_judge(bf, quest, judge_client)
        score_with_judge(bn, quest, judge_client)
        score_with_judge(gn, quest, judge_client)
        print("    full={:.2f}  nav={:.2f}  gkg={:.2f}".format(
            max(bf.judge_score, 0), max(bn.judge_score, 0), max(gn.judge_score, 0)))

    return bf, bn, gn


# ── GKG graph cache ───────────────────────────────────────────────────────────

_GKG_CACHE_PATH = "_gkg_cache.pkl"


def load_or_build_gkg(repo_path: str, cache_path: str = _GKG_CACHE_PATH):
    """Load a pickled GKG graph or build it fresh and cache it.

    Building via map_project is deterministic (pure AST parse, no LLM) but
    slow on large repos. Caching ensures repeated eval runs are identical and
    fast. Delete the cache file to force a rebuild.
    """
    p = Path(cache_path)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)

    from ast_mapper import map_project
    g = map_project(repo_path)
    with open(p, "wb") as f:
        pickle.dump(g, f)
    return g
