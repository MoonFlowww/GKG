# ============================================================
# gkg.py — core graph. 3 zooms (MACRO/MESO/MICRO), call-order, ownership, patterns.
# NANO disabled — coder agent reads full GKG as blueprint via tools.
# Pure. No network. No LLM.
# ============================================================

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Union, Any
import hashlib
import json
import uuid


# ── enums ───────────────────────────────────────────────────
class Level(str, Enum):
    MACRO = "MACRO"
    MESO  = "MESO"
    MICRO = "MICRO"


class Status(str, Enum):
    CONTESTED    = "CONTESTED"     # rank 0, frozen
    SKETCH       = "SKETCH"
    DESIGNED     = "DESIGNED"
    LAYER_STABLE = "LAYER_STABLE"
    READY        = "READY"
    CODED        = "CODED"


STATUS_RANK = {
    Status.CONTESTED:    0,
    Status.SKETCH:       1,
    Status.DESIGNED:     2,
    Status.LAYER_STABLE: 3,
    Status.READY:        4,
    Status.CODED:        5,
}

STATUS_ORDER = [Status.SKETCH, Status.DESIGNED, Status.LAYER_STABLE, Status.READY, Status.CODED]


class Phase(str, Enum):
    MACRO_DESIGN  = "MACRO_DESIGN"
    MACRO_STABLE  = "MACRO_STABLE"
    MESO_DESIGN   = "MESO_DESIGN"
    MESO_STABLE   = "MESO_STABLE"
    MICRO_DESIGN  = "MICRO_DESIGN"
    MICRO_STABLE  = "MICRO_STABLE"
    CODING        = "CODING"
    DONE          = "DONE"


PHASE_ORDER = list(Phase)
PHASE_RANK = {p: i for i, p in enumerate(PHASE_ORDER)}


class EdgeKind(str, Enum):
    OWN        = "OWN"
    CALLS      = "CALLS"
    SEND       = "SEND"
    IMPLEMENTS = "IMPLEMENTS"
    DEPENDS_ON = "DEPENDS_ON"


class Ownership(str, Enum):
    SINGLE_WRITER    = "SINGLE_WRITER"
    MULTI_SYNCED     = "MULTI_SYNCED"
    SHARED_IMMUTABLE = "SHARED_IMMUTABLE"
    PARTITIONED      = "PARTITIONED"


class DesignPattern(str, Enum):
    PAT_NONE   = "PAT_NONE"
    SINGLETON  = "SINGLETON"
    FACTORY    = "FACTORY"
    BUILDER    = "BUILDER"
    ADAPTER    = "ADAPTER"
    DECORATOR  = "DECORATOR"
    PROXY      = "PROXY"
    COMPOSITE  = "COMPOSITE"
    OBSERVER   = "OBSERVER"
    STATE      = "STATE"
    STRATEGY   = "STRATEGY"
    ITERATOR   = "ITERATOR"
    VISITOR    = "VISITOR"
    COMMAND    = "COMMAND"
    REPOSITORY = "REPOSITORY"


# ── payloads ────────────────────────────────────────────────
@dataclass
class MacroPayload:
    language: str = "python"
    ownership: Ownership = Ownership.SINGLE_WRITER


@dataclass
class MesoPayload:
    design_pattern: DesignPattern = DesignPattern.PAT_NONE
    behaviors: list[str] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0


@dataclass
class MicroPayload:
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0


Payload = Union[MacroPayload, MesoPayload, MicroPayload]


# ── node + edge ─────────────────────────────────────────────
@dataclass
class Node:
    id: str
    name: str
    level: Level
    status: Status
    intent: str
    parent: Optional[str]
    children: list[str] = field(default_factory=list)
    parent_fp: str = ""
    payload: Payload = None  # type: ignore


@dataclass
class Edge:
    id: str
    kind: EdgeKind
    src: str
    dst: str
    order: int = -1   # -1 = unordered


@dataclass
class Violation:
    rule: str
    at: str
    detail: str


# ── graph ───────────────────────────────────────────────────
class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self.phase: Phase = Phase.MACRO_DESIGN

    # ── fingerprint ───────────────────────────────────────
    @staticmethod
    def _canon(x: Any) -> str:
        s = json.dumps(x, sort_keys=True, default=_json_default)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def node_fp(self, n: Node) -> str:
        return self._canon({"name": n.name, "intent": n.intent, "payload": asdict(n.payload)})

    # ── creation ──────────────────────────────────────────
    def _add(self, n: Node) -> None:
        if n.id in self.nodes:
            raise ValueError(f"duplicate node id {n.id}")
        self.nodes[n.id] = n
        if n.parent is not None:
            self.nodes[n.parent].children.append(n.id)

    def add_macro(self, *, name: str, intent: str, language: str = "python",
                  ownership: Ownership = Ownership.SINGLE_WRITER) -> str:
        if self.phase != Phase.MACRO_DESIGN:
            raise ValueError(f"cannot add MACRO in phase {self.phase.value}. "
                             f"Reset graph or you are past the MACRO_DESIGN window.")
        nid = _newid()
        p = MacroPayload(language=language, ownership=ownership)
        self._add(Node(id=nid, name=name, level=Level.MACRO, status=Status.SKETCH,
                       intent=intent, parent=None, payload=p))
        return nid

    def add_meso(self, *, parent: str, name: str, intent: str,
                 design_pattern: DesignPattern = DesignPattern.PAT_NONE,
                 behaviors: Optional[list[str]] = None,
                 line_start: int = 0, line_end: int = 0) -> str:
        if PHASE_RANK[self.phase] < PHASE_RANK[Phase.MACRO_STABLE]:
            raise ValueError(
                f"cannot add MESO in phase {self.phase.value}. Required: MACRO_STABLE or later. "
                f"Promote all MACRO nodes to DESIGNED, then advance_phase to=MACRO_STABLE.")
        if parent not in self.nodes:
            raise ValueError(f"parent {parent} not found")
        par = self.nodes[parent]
        if par.level != Level.MACRO:
            raise ValueError(f"Meso parent must be MACRO, got {par.level.value}")
        nid = _newid()
        p = MesoPayload(design_pattern=design_pattern,
                        behaviors=list(behaviors or []),
                        line_start=line_start, line_end=line_end)
        self._add(Node(id=nid, name=name, level=Level.MESO, status=Status.SKETCH,
                       intent=intent, parent=parent,
                       parent_fp=self.node_fp(par), payload=p))
        return nid

    def add_micro(self, *, parent: str, name: str, intent: str,
                  inputs: Optional[list[str]] = None,
                  outputs: Optional[list[str]] = None,
                  line_start: int = 0, line_end: int = 0) -> str:
        if PHASE_RANK[self.phase] < PHASE_RANK[Phase.MESO_STABLE]:
            raise ValueError(
                f"cannot add MICRO in phase {self.phase.value}. Required: MESO_STABLE or later. "
                f"Promote all MESO nodes to DESIGNED, then advance_phase to=MESO_STABLE.")
        if parent not in self.nodes:
            raise ValueError(f"parent {parent} not found")
        par = self.nodes[parent]
        if par.level != Level.MESO:
            raise ValueError(f"Micro parent must be MESO, got {par.level.value}")
        nid = _newid()
        p = MicroPayload(inputs=list(inputs or []), outputs=list(outputs or []),
                         line_start=line_start, line_end=line_end)
        self._add(Node(id=nid, name=name, level=Level.MICRO, status=Status.SKETCH,
                       intent=intent, parent=parent,
                       parent_fp=self.node_fp(par), payload=p))
        return nid

    def add_edge(self, src: str, dst: str, kind: EdgeKind, *, order: int = -1) -> str:
        if src not in self.nodes:
            raise ValueError(f"edge src {src} not found")
        if dst not in self.nodes:
            raise ValueError(f"edge dst {dst} not found")
        eid = _newid()
        self.edges[eid] = Edge(id=eid, kind=kind, src=src, dst=dst, order=order)
        return eid

    # ── status machine ────────────────────────────────────
    def _promote_one(self, nid: str, to: Status) -> None:
        if nid not in self.nodes:
            raise ValueError(f"node {nid} not found")
        n = self.nodes[nid]
        if n.status == Status.CONTESTED:
            raise ValueError("node is CONTESTED; resolve first")
        cur = STATUS_RANK[n.status]
        nxt = STATUS_RANK[to]
        if nxt != cur + 1:
            raise ValueError(f"internal: _promote_one called non-adjacent {n.status.value} → {to.value}")

        if to == Status.LAYER_STABLE:
            if n.parent is None:
                sibs = [m for m in self.nodes.values()
                        if m.level == n.level and m.parent is None]
            else:
                sibs = [self.nodes[c] for c in self.nodes[n.parent].children]
            laggards = [s for s in sibs if STATUS_RANK[s.status] < STATUS_RANK[Status.DESIGNED]]
            if laggards:
                names = ", ".join(f"{s.name}({s.status.value})" for s in laggards)
                raise ValueError(
                    f"cannot reach LAYER_STABLE: siblings not yet DESIGNED: {names}. "
                    f"Use promote_group with all sibling ids to move them together.")

        elif to == Status.READY:
            stuck = [self.nodes[c] for c in n.children
                     if STATUS_RANK[self.nodes[c].status] < STATUS_RANK[Status.LAYER_STABLE]]
            if stuck:
                names = ", ".join(f"{s.name}({s.status.value})" for s in stuck)
                raise ValueError(
                    f"cannot reach READY: children not yet LAYER_STABLE: {names}. "
                    f"Promote children up first.")

        n.status = to

    def promote(self, nid: str, target: Status) -> list[Status]:
        """Auto-chains SKETCH → ... → target. Idempotent."""
        if nid not in self.nodes:
            raise ValueError(f"node {nid} not found")
        n = self.nodes[nid]
        if n.status == Status.CONTESTED:
            raise ValueError("node is CONTESTED; resolve first")
        if target == Status.CONTESTED:
            raise ValueError("cannot promote TO CONTESTED")
        walked: list[Status] = []
        while STATUS_RANK[n.status] < STATUS_RANK[target]:
            cur_rank = STATUS_RANK[n.status]
            next_status = next(s for s in STATUS_ORDER if STATUS_RANK[s] == cur_rank + 1)
            self._promote_one(nid, next_status)
            walked.append(next_status)
        return walked

    def promote_group(self, ids: list[str], target: Status) -> dict[str, list[Status]]:
        """Sibling-aware batch promote. All ids climb in lockstep, one rank at a time."""
        if target == Status.CONTESTED:
            raise ValueError("cannot promote TO CONTESTED")
        if not ids:
            return {}
        for i in ids:
            if i not in self.nodes:
                raise ValueError(f"node {i} not found")
            if self.nodes[i].status == Status.CONTESTED:
                raise ValueError(f"node {self.nodes[i].name} is CONTESTED")
        walked: dict[str, list[Status]] = {i: [] for i in ids}
        target_rank = STATUS_RANK[target]
        while any(STATUS_RANK[self.nodes[i].status] < target_rank for i in ids):
            for i in ids:
                n = self.nodes[i]
                cur_rank = STATUS_RANK[n.status]
                if cur_rank >= target_rank:
                    continue
                nxt = next(s for s in STATUS_ORDER if STATUS_RANK[s] == cur_rank + 1)
                self._promote_one(i, nxt)
                walked[i].append(nxt)
        return walked

    # ── phase machine ─────────────────────────────────────
    def _advance_one(self, target: Phase) -> None:
        ci = PHASE_ORDER.index(self.phase)
        ti = PHASE_ORDER.index(target)
        if ti != ci + 1:
            raise ValueError(f"internal: non-sequential {self.phase.value} → {target.value}")
        needs = {Phase.MACRO_STABLE: Level.MACRO, Phase.MESO_STABLE: Level.MESO,
                 Phase.MICRO_STABLE: Level.MICRO}
        if target in needs:
            lvl = needs[target]
            if not any(n.level == lvl for n in self.nodes.values()):
                raise ValueError(f"cannot reach {target.value}: no {lvl.value} nodes exist yet. "
                                 f"Add some with add_{lvl.value.lower()} first.")
            stuck = [n for n in self.nodes.values()
                     if n.level == lvl and STATUS_RANK[n.status] < STATUS_RANK[Status.DESIGNED]]
            if stuck:
                names = ", ".join(f"{n.name}({n.status.value})" for n in stuck)
                raise ValueError(f"cannot reach {target.value}: these {lvl.value} nodes need "
                                 f"DESIGNED first: {names}. Call promote_group on them.")
        self.phase = target

    def advance_phase(self, target: Phase) -> list[Phase]:
        """Walks phase forward every step until reaching target. Idempotent."""
        if PHASE_RANK[target] < PHASE_RANK[self.phase]:
            raise ValueError(f"cannot move backwards: current {self.phase.value}, target {target.value}")
        walked: list[Phase] = []
        while self.phase != target:
            next_rank = PHASE_RANK[self.phase] + 1
            nxt = next(p for p in PHASE_ORDER if PHASE_RANK[p] == next_rank)
            self._advance_one(nxt)
            walked.append(nxt)
        return walked

    # ── validators ────────────────────────────────────────
    def validate(self) -> list[Violation]:
        out: list[Violation] = []
        out += self._rule_inheritance()
        out += self._rule_design_pattern()
        out += self._rule_call_order()
        out += self._rule_ownership()
        out += self._rule_orphan_macro()
        return out

    def _rule_inheritance(self) -> list[Violation]:
        out = []
        for n in self.nodes.values():
            if n.parent is None:
                continue
            par = self.nodes[n.parent]
            cur = self.node_fp(par)
            if cur != n.parent_fp:
                out.append(Violation("D_FP", n.id,
                    f"parent fp stale: have {n.parent_fp[:8]}, want {cur[:8]}"))
        return out

    def _rule_design_pattern(self) -> list[Violation]:
        out = []
        for n in self.nodes.values():
            if n.level != Level.MESO:
                continue
            p = n.payload
            assert isinstance(p, MesoPayload)
            if p.design_pattern == DesignPattern.PAT_NONE:
                continue
            req = PAT_REQ.get(p.design_pattern)
            if req is None:
                continue
            names_lower = [b.lower() for b in p.behaviors]
            for r in req:
                if not any(r in nm for nm in names_lower):
                    out.append(Violation("J_BEH", n.id,
                        f"{p.design_pattern.value} needs behavior containing '{r}'"))
        return out

    def _rule_call_order(self) -> list[Violation]:
        out = []
        by_src: dict[str, list[Edge]] = {}
        for e in self.edges.values():
            if e.kind != EdgeKind.CALLS or e.order < 0:
                continue
            by_src.setdefault(e.src, []).append(e)
        for src, es in by_src.items():
            es_sorted = sorted(es, key=lambda x: x.order)
            ords = [e.order for e in es_sorted]
            if ords != list(range(1, len(ords) + 1)):
                out.append(Violation("K_ORDER", src, f"call order {ords} not 1..n"))
        return out

    def _rule_ownership(self) -> list[Violation]:
        out = []
        counts: dict[str, int] = {}
        for e in self.edges.values():
            if e.kind == EdgeKind.OWN:
                counts[e.dst] = counts.get(e.dst, 0) + 1
        for dst, c in counts.items():
            if c > 1:
                out.append(Violation("O_MULTI_OWNER", dst, f"{c} owners (must be 1)"))
        return out

    def _rule_orphan_macro(self) -> list[Violation]:
        out = []
        if PHASE_RANK[self.phase] >= PHASE_RANK[Phase.MESO_STABLE]:
            for n in self.nodes.values():
                if n.level == Level.MACRO and not n.children:
                    out.append(Violation("V_ORPHAN_MACRO", n.id,
                        f"MACRO '{n.name}' has no MESO children after MESO_STABLE"))
        return out

    # ── persistence ───────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "phase": self.phase.value,
            "nodes": [_node_to_dict(n) for n in self.nodes.values()],
            "edges": [asdict(e, dict_factory=_enum_dict_factory) for e in self.edges.values()],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── design-pattern required behaviors (keywords that must appear in behaviors list) ──
PAT_REQ: dict[DesignPattern, list[str]] = {
    DesignPattern.SINGLETON:  ["instance"],
    DesignPattern.FACTORY:    ["create"],
    DesignPattern.BUILDER:    ["build"],
    DesignPattern.ADAPTER:    [],
    DesignPattern.DECORATOR:  [],
    DesignPattern.PROXY:      [],
    DesignPattern.COMPOSITE:  ["add", "remove"],
    DesignPattern.OBSERVER:   ["notify"],
    DesignPattern.STATE:      ["transition"],
    DesignPattern.ITERATOR:   ["next"],
    DesignPattern.VISITOR:    ["visit"],
    DesignPattern.COMMAND:    ["execute"],
    DesignPattern.REPOSITORY: ["find", "save"],
}



# ── helpers ─────────────────────────────────────────────────
def _newid() -> str:
    return str(uuid.uuid4())


def _json_default(o: Any) -> Any:
    if isinstance(o, Enum):
        return o.value
    raise TypeError(f"not JSON-serializable: {type(o)}")


def _enum_dict_factory(items):
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in items}


def _node_to_dict(n: Node) -> dict:
    d = asdict(n, dict_factory=_enum_dict_factory)
    return d
