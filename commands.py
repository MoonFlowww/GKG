# ============================================================
# commands.py — bridge between LLM and GKG.
# LLM emits JSON → dispatch → GKG mutation → result fed back.
# NANO disabled — coder agent reads full GKG blueprint via tools instead.
# ============================================================

from __future__ import annotations
import json
from typing import Any, Optional
from dataclasses import asdict
from enum import Enum

from gkg import (
    Graph, Node, Level, Status, Phase, EdgeKind, Ownership, DesignPattern,
    MacroPayload, MesoPayload, MicroPayload,
    Violation, STATUS_RANK, PHASE_RANK,
)
from ollama_client import OllamaClient


# ── system prompt ───────────────────────────────────────────
COMMAND_SYS = """You are an architect agent building a Generalized Knowledge Graph (GKG).
Three zoom levels: MACRO (modules), MESO (classes/patterns), MICRO (functions).
The GKG is a blueprint — a separate coder agent will read it via tools to write code.
Keep intent and contracts precise and minimal.

You act ONLY by emitting ONE JSON command per turn. No prose. No markdown.

Schema: { "cmd": "<name>", "args": { ... } }

COMMANDS:

  list_nodes        args: {}
                    -> [{id, name, level, status, parent}]

  list_edges        args: {}
                    -> [{id, kind, src, dst, order}]

  inspect           args: {"id": <full_id>}

  add_macro         args: {"name", "intent",
                           "language" (default "python"),
                           "ownership" (SINGLE_WRITER|MULTI_SYNCED|SHARED_IMMUTABLE|PARTITIONED)}

  add_meso          args: {"parent" <macro_id>, "name", "intent",
                           "design_pattern" (PAT_NONE|SINGLETON|FACTORY|BUILDER|ADAPTER|
                                             DECORATOR|PROXY|COMPOSITE|OBSERVER|STATE|
                                             STRATEGY|ITERATOR|VISITOR|COMMAND|REPOSITORY),
                           "behaviors" [str]}

  add_micro         args: {"parent" <meso_id>, "name", "intent",
                           "inputs" [str], "outputs" [str]}

  add_edge          args: {"src", "dst",
                           "kind" (OWN|CALLS|SEND|IMPLEMENTS|DEPENDS_ON),
                           "order" int (-1 if unordered)}

  promote           args: {"id", "to" (DESIGNED|LAYER_STABLE|READY)}
                    AUTO-CHAINS. Idempotent.
                    WARNING: to reach LAYER_STABLE, ALL siblings must be DESIGNED.
                    If you have multiple siblings, use promote_group.

  promote_group     args: {"ids": [id1, id2, ...], "to": <status>}
                    Sibling-aware. Walks all ids rank by rank together.
                    USE WHEN promoting multiple nodes of the same level/parent.

  advance_phase     args: {"to" <any later phase>}
                    AUTO-CHAINS through every intermediate phase.

  validate          args: {}
                    -> [{rule, at, detail}]

  done              args: {"reason": str}
                    ends the loop. Call after validate returns no violations.

TYPICAL WORKFLOW:
  1. add_macro for each module
  2. promote_group ids=[all macro ids] to=LAYER_STABLE
  3. advance_phase to=MACRO_STABLE
  4. add_meso under each macro
  5. promote_group sibling meso ids to=LAYER_STABLE
  6. advance_phase to=MESO_STABLE
  7. add_micro under each meso → promote_group → advance_phase to=MICRO_STABLE
  8. promote micro nodes to=READY (bottom-up)
  9. validate
  10. done

RULES:
  - Every reply = ONE JSON object.
  - NEVER repeat the same failing command. Read the error — it tells you what to do.
  - Use promote_group for siblings, not promote one at a time.
"""


# ── helpers ─────────────────────────────────────────────────
def _parse_enum(enum_cls, s: str):
    up = str(s).upper()
    for member in enum_cls:
        if member.value == up:
            return member
    valid = "|".join(m.value for m in enum_cls)
    raise ValueError(f"bad {enum_cls.__name__} '{s}'. valid: {valid}")


def _strs(x) -> list[str]:
    return [] if x is None else [str(s) for s in x]


# ── dispatch ────────────────────────────────────────────────
def dispatch(g: Graph, cmd: str, args: dict) -> dict:
    """Execute one command. Always returns {ok, result} or {ok:false, error}."""
    try:
        return {"ok": True, "result": _exec(g, cmd, args or {})}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _exec(g: Graph, cmd: str, a: dict) -> Any:
    if cmd == "list_nodes":
        return [{"id": n.id, "name": n.name, "level": n.level.value,
                 "status": n.status.value, "parent": n.parent}
                for n in g.nodes.values()]

    elif cmd == "list_edges":
        return [{"id": e.id, "kind": e.kind.value, "src": e.src, "dst": e.dst,
                 "order": e.order} for e in g.edges.values()]

    elif cmd == "inspect":
        if a["id"] not in g.nodes:
            raise ValueError(f"node {a['id']} not found")
        n = g.nodes[a["id"]]
        return {"id": n.id, "name": n.name, "level": n.level.value,
                "status": n.status.value, "intent": n.intent,
                "parent": n.parent, "children": n.children,
                "payload": asdict(n.payload, dict_factory=_enum_dict_factory)}

    elif cmd == "add_macro":
        return g.add_macro(
            name=a["name"], intent=a["intent"],
            language=a.get("language", "python"),
            ownership=_parse_enum(Ownership, a.get("ownership", "SINGLE_WRITER")))

    elif cmd == "add_meso":
        return g.add_meso(
            parent=a["parent"], name=a["name"], intent=a["intent"],
            design_pattern=_parse_enum(DesignPattern, a.get("design_pattern", "PAT_NONE")),
            behaviors=_strs(a.get("behaviors")))

    elif cmd == "add_micro":
        return g.add_micro(
            parent=a["parent"], name=a["name"], intent=a["intent"],
            inputs=_strs(a.get("inputs")),
            outputs=_strs(a.get("outputs")))

    elif cmd == "add_edge":
        return g.add_edge(a["src"], a["dst"],
                          _parse_enum(EdgeKind, a["kind"]),
                          order=int(a.get("order", -1)))

    elif cmd == "promote":
        walked = g.promote(a["id"], _parse_enum(Status, a["to"]))
        return f"promoted {a['id']} via {[s.value for s in walked]}"

    elif cmd == "promote_group":
        ids = list(a["ids"])
        target = _parse_enum(Status, a["to"])
        walked = g.promote_group(ids, target)
        total_steps = sum(len(v) for v in walked.values())
        return f"promoted {len(ids)} nodes, {total_steps} total rank-steps, all → {target.value}"

    elif cmd == "advance_phase":
        walked = g.advance_phase(_parse_enum(Phase, a["to"]))
        return f"phase = {g.phase.value} (walked {[p.value for p in walked]})"

    elif cmd == "validate":
        return [{"rule": v.rule, "at": v.at, "detail": v.detail} for v in g.validate()]

    elif cmd == "done":
        return "done"

    else:
        raise ValueError(f"unknown command '{cmd}'")


def _enum_dict_factory(items):
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in items}



# ── next-action hints ───────────────────────────────────────
def suggest_next(g: Graph) -> str:
    hints: list[str] = []

    # sibling groups of SKETCH nodes
    sibling_groups: dict[tuple, list[Node]] = {}
    for n in g.nodes.values():
        if n.status == Status.SKETCH:
            sibling_groups.setdefault((n.level, n.parent), []).append(n)

    for (lvl, _par), group in sibling_groups.items():
        if len(group) == 1:
            hints.append(f"SKETCH {lvl.value} '{group[0].name}': "
                         f"call promote id={group[0].id} to=LAYER_STABLE")
        else:
            ids = [n.id for n in group]
            names = ", ".join(n.name for n in group)
            hints.append(f"SKETCH {lvl.value} siblings [{names}]: "
                         f"call promote_group ids={json.dumps(ids)} to=LAYER_STABLE")

    phase = g.phase
    levels = {lvl: [n for n in g.nodes.values() if n.level == lvl] for lvl in Level}

    def all_at(lvl: Level, rank: int) -> bool:
        return bool(levels[lvl]) and all(STATUS_RANK[n.status] >= rank for n in levels[lvl])

    stable = STATUS_RANK[Status.LAYER_STABLE]

    if phase == Phase.MACRO_DESIGN and all_at(Level.MACRO, stable):
        hints.append("All MACRO LAYER_STABLE. Call advance_phase to=MACRO_STABLE.")
    if PHASE_RANK[phase] >= PHASE_RANK[Phase.MACRO_STABLE] and levels[Level.MACRO] and not levels[Level.MESO]:
        hints.append("Phase allows MESO. Call add_meso with parent=<macro_id>.")
    if phase == Phase.MESO_DESIGN and all_at(Level.MESO, stable):
        hints.append("All MESO LAYER_STABLE. Call advance_phase to=MESO_STABLE.")
    if PHASE_RANK[phase] >= PHASE_RANK[Phase.MESO_STABLE] and levels[Level.MESO] and not levels[Level.MICRO]:
        hints.append("Phase allows MICRO. Call add_micro with parent=<meso_id>.")
    if phase == Phase.MICRO_DESIGN and all_at(Level.MICRO, stable):
        hints.append("All MICRO LAYER_STABLE. Call advance_phase to=MICRO_STABLE.")
    if phase == Phase.MICRO_STABLE and all_at(Level.MICRO, STATUS_RANK[Status.READY]):
        hints.append("All MICRO READY. Call validate then done.")

    return "\n  - " + "\n  - ".join(hints) if hints else "(follow workflow)"


def describe_for_coder(g: Graph) -> str:
    """Blueprint summary for coder agent. Exposes intent + full payload contracts."""
    lines = ["# GKG Blueprint"]
    macros = [n for n in g.nodes.values() if n.level == Level.MACRO]
    for macro in macros:
        assert isinstance(macro.payload, MacroPayload)
        lines.append(f"\n## MODULE: {macro.name}")
        lines.append(f"   intent: {macro.intent}")
        lines.append(f"   language: {macro.payload.language}  ownership: {macro.payload.ownership.value}")
        mesos = [g.nodes[c] for c in macro.children]
        for meso in mesos:
            assert isinstance(meso.payload, MesoPayload)
            pat = meso.payload.design_pattern.value
            lines.append(f"\n   CLASS: {meso.name}  [pattern={pat}]")
            lines.append(f"   intent: {meso.intent}")
            if meso.payload.behaviors:
                lines.append(f"   behaviors: {meso.payload.behaviors}")
            micros = [g.nodes[c] for c in meso.children]
            for micro in micros:
                assert isinstance(micro.payload, MicroPayload)
                lines.append(f"\n      METHOD: {micro.name}")
                lines.append(f"      intent: {micro.intent}")
                if micro.payload.inputs:
                    lines.append(f"      inputs:  {micro.payload.inputs}")
                if micro.payload.outputs:
                    lines.append(f"      outputs: {micro.payload.outputs}")
    if g.edges:
        lines.append("\n# CROSS-MODULE DEPENDENCIES")
        for e in g.edges.values():
            sn = g.nodes[e.src].name if e.src in g.nodes else e.src
            dn = g.nodes[e.dst].name if e.dst in g.nodes else e.dst
            order_str = f" (call_order={e.order})" if e.order >= 0 else ""
            lines.append(f"   {sn} --[{e.kind.value}]--> {dn}{order_str}")
    return "\n".join(lines)


def describe_state(g: Graph) -> str:
    lines = [
        f"phase = {g.phase.value}",
        f"counts: nodes={len(g.nodes)} edges={len(g.edges)}",
        "",
        "NODES:",
    ]
    if not g.nodes:
        lines.append("  (none)")
    else:
        for n in g.nodes.values():
            lines.append(f"  [{n.level.value}] id={n.id}")
            lines.append(f"    name={n.name} status={n.status.value} parent={n.parent}")

    lines += ["", "EDGES:"]
    if not g.edges:
        lines.append("  (none)")
    else:
        for e in g.edges.values():
            lines.append(f"  {e.kind.value}: {e.src} -> {e.dst} order={e.order}")

    lines += ["", "NEXT LEGAL ACTIONS:", f"  {suggest_next(g)}"]
    return "\n".join(lines)


# ── agent loop ──────────────────────────────────────────────
def run_agent(g: Graph, client: OllamaClient, goal: str, *,
              max_steps: int = 50,
              repeat_abort: int = 3,
              verbose: bool = True) -> list[dict]:
    """Run the architect command loop. Returns list of per-step trace entries."""
    last_result: dict = {"ok": True, "result": "(start)"}
    trace: list[dict] = []
    last_fail_sig = ""
    fail_streak = 0

    for step in range(1, max_steps + 1):
        prompt = (
            f"GOAL: {goal}\n\n"
            f"CURRENT STATE:\n{describe_state(g)}\n\n"
            f"LAST RESULT: {json.dumps(last_result)}\n\n"
            f"Emit ONE JSON command now."
        )

        try:
            raw = client.complete_json(prompt, system=COMMAND_SYS)
        except Exception as e:
            if verbose:
                print(f"[{step}] BAD JSON: {e}")
            last_result = {"ok": False,
                "error": "your previous output was not valid JSON. "
                         "Emit one JSON object with 'cmd' and 'args' fields only."}
            trace.append({"step": step, "cmd": None, "error": str(e)})
            continue

        cmd = str(raw.get("cmd", ""))
        args = raw.get("args", {}) or {}

        if not cmd:
            if verbose:
                print(f"[{step}] missing 'cmd'")
            last_result = {"ok": False, "error": "missing 'cmd' field"}
            trace.append({"step": step, "cmd": None, "error": "missing cmd"})
            continue

        if verbose:
            print(f"[{step}] {cmd} ", end="", flush=True)

        result = dispatch(g, cmd, args)
        last_result = result
        trace.append({"step": step, "cmd": cmd, "args": args, "result": result})

        if result["ok"]:
            if verbose:
                print("✓")
            fail_streak = 0
            last_fail_sig = ""
        else:
            err = str(result["error"])
            if verbose:
                print(f"✗ {err}")
            sig = f"{cmd}|{err[:40]}"
            if sig == last_fail_sig:
                fail_streak += 1
                if fail_streak >= repeat_abort:
                    if verbose:
                        print(f"\n✗ ABORT: same failure {repeat_abort}×. Last error: {err}")
                    trace.append({"step": step, "aborted": True})
                    return trace
                last_result = {"ok": False,
                    "error": f"STOP. '{cmd}' failed {fail_streak} times with same error. "
                             f"DO NOT RETRY '{cmd}'. Read CURRENT STATE and NEXT LEGAL ACTIONS. "
                             f"Pick a DIFFERENT command. Original error: {err}"}
            else:
                fail_streak = 1
                last_fail_sig = sig

        if cmd == "done":
            return trace

    if verbose:
        print(f"\nhit max_steps={max_steps}")
    return trace
