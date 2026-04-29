from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from gkg import Graph
from ollama_client import OllamaClient
from ast_mapper import map_project
from designer import GKGBlueprint, design_feature
from implementer import implement_feature


@dataclass
class PipelineResult:
    graph: Graph
    blueprint: GKGBlueprint
    diffs: dict[str, str] = field(default_factory=dict)
    new_contents: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"nodes mapped:    {len(self.graph.nodes)}",
            f"edges mapped:    {len(self.graph.edges)}",
            f"blueprint nodes: {len(self.blueprint.nodes)}",
            f"files changed:   {len(self.diffs)}",
        ]
        for fpath, diff in self.diffs.items():
            lines.append(f"  {fpath}: {diff.count(chr(10))} diff lines")
        return "\n".join(lines)

    def print_diffs(self) -> None:
        for fpath, diff in self.diffs.items():
            print(f"\n{'='*60}\n{fpath}\n{'='*60}")
            print(diff)


def run(project_path: str, feature: str, client: OllamaClient) -> PipelineResult:
    print(f"[1/3] mapping {project_path!r}")
    g = map_project(project_path, client)
    print(f"      {len(g.nodes)} nodes, {len(g.edges)} edges")

    print(f"[2/3] designing {feature!r}")
    bp = design_feature(g, feature, client)
    print(f"      {len(bp.nodes)} blueprint nodes")
    print(f"      {bp.summary}")

    print("[3/3] implementing")
    diffs, new_contents = implement_feature(g, bp, project_path, client)
    print(f"      {len(diffs)} file(s) with changes")

    return PipelineResult(graph=g, blueprint=bp, diffs=diffs, new_contents=new_contents)


def apply_changes(project_path: str, new_contents: dict[str, str]) -> None:
    """Write new file contents to disk. Use new_contents from PipelineResult, not diffs."""
    root = Path(project_path)
    for rel_path, content in new_contents.items():
        fpath = root / rel_path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content, encoding="utf-8")
        print(f"  wrote: {rel_path}")
