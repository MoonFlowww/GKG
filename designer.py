from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict

from gkg import Graph
from ollama_client import OllamaClient
from commands import describe_for_coder


@dataclass
class BpNode:
    action: str           # "add" | "modify"
    level: str            # "MACRO" | "MESO" | "MICRO"
    name: str
    parent_name: str      # parent node name in graph (for add); existing node name (for modify)
    intent: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    behaviors: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class BpEdge:
    src_name: str
    dst_name: str
    kind: str


@dataclass
class GKGBlueprint:
    feature: str
    nodes: list[BpNode] = field(default_factory=list)
    edges: list[BpEdge] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        lines = [f"# Blueprint: {self.feature}", f"\n{self.summary}\n", "## Nodes:"]
        for n in self.nodes:
            lines.append(f"  [{n.action.upper()}] {n.level} '{n.name}' parent={n.parent_name!r}")
            lines.append(f"    intent: {n.intent}")
            if n.notes:
                lines.append(f"    notes:  {n.notes}")
        if self.edges:
            lines.append("\n## Edges:")
            for e in self.edges:
                lines.append(f"  {e.src_name} --[{e.kind}]--> {e.dst_name}")
        return "\n".join(lines)


_DESIGNER_SYS = """\
You are a software architect. Given a codebase knowledge graph and a feature request,
produce a blueprint describing exactly what to add or modify.

Output ONE JSON object:
{
  "feature": "<name>",
  "summary": "<2-3 sentence design rationale>",
  "nodes": [
    {
      "action": "add" | "modify",
      "level": "MACRO" | "MESO" | "MICRO",
      "name": "<short name, no path>",
      "parent_name": "<existing node name from graph; empty for MACRO add>",
      "intent": "<one sentence>",
      "inputs": ["<arg>", ...],
      "outputs": ["<type>", ...],
      "behaviors": ["<behavior>", ...],
      "notes": "<exact impl guidance for the coder>"
    }
  ],
  "edges": [
    { "src_name": "<name>", "dst_name": "<name>", "kind": "CALLS|DEPENDS_ON|IMPLEMENTS" }
  ]
}

Rules:
- Only describe what the feature requires. No extras.
- "modify": parent_name = name of the node to change.
- "add": parent_name must match an existing node name in the graph.
- inputs/outputs only for MICRO. behaviors only for MESO.
- notes must tell the coder exactly what the code must do — be specific.
"""


def design_feature(g: Graph, feature: str, client: OllamaClient) -> GKGBlueprint:
    gkg_text = describe_for_coder(g)
    prompt = (
        f"CODEBASE GKG:\n{gkg_text}\n\n"
        f"FEATURE TO IMPLEMENT:\n{feature}\n\n"
        "Emit ONE JSON blueprint now."
    )
    raw = client.complete_json(prompt, system=_DESIGNER_SYS, max_tokens=4000, label="design")

    nodes = [
        BpNode(
            action=n.get("action", "add"),
            level=n.get("level", "MICRO"),
            name=n.get("name", ""),
            parent_name=n.get("parent_name", ""),
            intent=n.get("intent", ""),
            inputs=list(n.get("inputs") or []),
            outputs=list(n.get("outputs") or []),
            behaviors=list(n.get("behaviors") or []),
            notes=n.get("notes", ""),
        )
        for n in (raw.get("nodes") or [])
    ]
    edges = [
        BpEdge(
            src_name=e.get("src_name", ""),
            dst_name=e.get("dst_name", ""),
            kind=e.get("kind", "CALLS"),
        )
        for e in (raw.get("edges") or [])
    ]
    return GKGBlueprint(
        feature=raw.get("feature", feature),
        nodes=nodes,
        edges=edges,
        summary=raw.get("summary", ""),
    )
