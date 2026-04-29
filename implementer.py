from __future__ import annotations
import difflib
import json
from pathlib import Path
from typing import Optional

from gkg import Graph
from ollama_client import OllamaClient
from commands import describe_for_coder
from designer import GKGBlueprint, BpNode


_IMPL_SYS = """\
You are a code implementer. Given a blueprint node and the current file content,
produce the updated file.

Output ONE JSON object:
{
  "file": "<relative file path>",
  "content": "<complete new file content>"
}

Rules:
- For "add": insert the new class/function in the appropriate file.
- For "modify": return the entire updated file, not just the changed part.
- Preserve all existing code. Only change what the blueprint specifies.
- Output valid, runnable code. No placeholders or TODO stubs.
- "content" is the complete file as a string (use \\n for newlines).
"""


def _read(project_path: str, rel_path: str) -> str:
    p = Path(project_path) / rel_path
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def _unified_diff(old: str, new: str, filename: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    ))


def _resolve_file(node: BpNode, g: Graph) -> str:
    """Best-effort: determine which file should contain this node."""
    # parent_name may be a qualified name like "foo/bar.py::ClassName"
    if "::" in node.parent_name:
        return node.parent_name.split("::")[0]
    # look for a graph node matching parent_name
    for n in g.nodes.values():
        if n.name == node.parent_name or n.name.endswith(f"::{node.parent_name}"):
            # walk up to MACRO (file)
            cur = n
            while cur.parent:
                cur = g.nodes[cur.parent]
            return cur.name  # MACRO name = rel_path
    # fallback
    if node.level == "MACRO":
        return node.name if node.name.endswith((".py", ".js", ".ts", ".go")) else f"{node.name}.py"
    return "new_module.py"


def implement_feature(
    g: Graph,
    bp: GKGBlueprint,
    project_path: str,
    client: OllamaClient,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns (diffs, new_contents).
      diffs:        {rel_path: unified_diff_string}  — for human review
      new_contents: {rel_path: full_file_content}    — for apply_changes()
    """
    gkg_summary = describe_for_coder(g)
    accumulated: dict[str, str] = {}  # rel_path → latest full content

    for node in bp.nodes:
        target_file = _resolve_file(node, g)
        current = accumulated.get(target_file) or _read(project_path, target_file)

        prompt = (
            f"GKG CONTEXT (summary):\n{gkg_summary[:3000]}\n\n"
            f"BLUEPRINT NODE:\n{json.dumps(_node_dict(node), indent=2)}\n\n"
            f"TARGET FILE: {target_file}\n"
            f"CURRENT CONTENT:\n{current or '(new file)'}\n\n"
            "Emit ONE JSON with 'file' and 'content' keys."
        )

        try:
            raw = client.complete_json(prompt, system=_IMPL_SYS, max_tokens=8000, label="implement")
            new_content = raw.get("content", "")
            resolved_file = raw.get("file", target_file)
            if new_content:
                accumulated[resolved_file] = new_content
        except Exception as e:
            print(f"[implementer] warning: AI failed for '{node.name}': {e}")

    diffs: dict[str, str] = {}
    for rel_path, new_content in accumulated.items():
        old_content = _read(project_path, rel_path)
        diff = _unified_diff(old_content, new_content, rel_path)
        if diff:
            diffs[rel_path] = diff

    return diffs, accumulated


def _node_dict(node: BpNode) -> dict:
    return {
        "action": node.action,
        "level": node.level,
        "name": node.name,
        "parent_name": node.parent_name,
        "intent": node.intent,
        "inputs": node.inputs,
        "outputs": node.outputs,
        "behaviors": node.behaviors,
        "notes": node.notes,
    }
