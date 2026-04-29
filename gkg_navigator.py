"""GKG Navigator — cd-style interface for AI agents to explore a mapped codebase.

Navigation model:
  - Default view : full MACRO dump (all modules, intent, child count)
  - cd <name>    : move into a MACRO -> dumps all MESOs; or into a MESO -> dumps all MICROs
  - edges [kind] : list edges from current node (horizontal navigation)
  - get_code <name> : return source snippet for a MESO (class body) or MICRO (method body)
  - up           : go one level up
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from gkg import Graph, Level, EdgeKind, Node


# ── source extraction ────────────────────────────────────────────────────────

def _load_ts_cpp():
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_cpp as m
        return Language(m.language()), Parser(Language(m.language()))
    except Exception:
        return None, None


def _extract_node_source(src: str, class_name: str, method_name: Optional[str] = None) -> str:
    """Return source snippet for a class or a method inside a class using tree-sitter."""
    lang, parser = _load_ts_cpp()
    if parser is None:
        return _extract_node_source_regex(src, class_name, method_name)

    tree = parser.parse(src.encode(errors="replace"))

    def _find_class(node):
        if node.type in ("class_specifier", "struct_specifier"):
            for c in node.children:
                if c.type == "type_identifier" and c.text.decode(errors="replace") == class_name:
                    return node
        for child in node.children:
            result = _find_class(child)
            if result:
                return result
        return None

    class_node = _find_class(tree.root_node)
    if class_node is None:
        return f"// class {class_name!r} not found in source"

    if method_name is None:
        return src[class_node.start_byte: class_node.end_byte]

    # find method inside field_declaration_list
    body_node = next((c for c in class_node.children if c.type == "field_declaration_list"), None)
    if body_node is None:
        return f"// no body found for {class_name}"

    def _method_name(node) -> str:
        for c in node.children:
            if c.type in ("function_declarator", "reference_declarator", "pointer_declarator"):
                return _method_name(c)
            if c.type in ("identifier", "field_identifier", "destructor_name", "operator_name"):
                return c.text.decode(errors="replace")
        return ""

    def _find_method(node):
        if node.type in ("function_definition", "declaration", "field_declaration"):
            if _method_name(node) == method_name:
                return node
        elif node.type == "template_declaration":
            for c in node.children:
                r = _find_method(c)
                if r:
                    return r
        return None

    for child in body_node.children:
        m_node = _find_method(child)
        if m_node:
            return src[m_node.start_byte: m_node.end_byte]

    return f"// method {method_name!r} not found in {class_name}"


def _extract_node_source_regex(src: str, class_name: str, method_name: Optional[str]) -> str:
    import re
    pat = re.compile(rf'\b(?:class|struct)\s+{re.escape(class_name)}\b')
    m = pat.search(src)
    if not m:
        return f"// {class_name} not found"
    depth = 0
    i = m.start()
    start = None
    while i < len(src):
        if src[i] == '{':
            if start is None:
                start = i
            depth += 1
        elif src[i] == '}':
            depth -= 1
            if depth == 0 and start is not None:
                return src[m.start(): i + 1]
        i += 1
    return f"// could not extract {class_name}"


# ── Navigator ────────────────────────────────────────────────────────────────

class GKGNavigator:
    """Stateful navigator over a GKG Graph.

    Designed for use inside AI agent loops — each method returns a compact
    text string the AI can read directly.
    """

    def __init__(self, graph: Graph, project_path: str):
        self.g = graph
        self.root = Path(project_path)
        self._current: Optional[str] = None   # node id; None = MACRO level
        self._source_cache: dict[str, str] = {}

        # index: name -> node_id  (short name, lower-cased for fuzzy lookup)
        self._name_index: dict[str, str] = {}
        for nid, node in graph.nodes.items():
            short = node.name.split("::")[-1].lower()
            self._name_index[short] = nid
            self._name_index[node.name.lower()] = nid

    # ── public API ───────────────────────────────────────────────────────────

    def dump(self) -> str:
        """Return the current level's full dump. Starting point for any session."""
        if self._current is None:
            return self._dump_macros()
        node = self.g.nodes[self._current]
        if node.level == Level.MACRO:
            return self._dump_meso_level(node)
        if node.level == Level.MESO:
            return self._dump_micro_level(node)
        return f"[{node.name}] — leaf node (MICRO). Use get_code to read source."

    def cd(self, name: str) -> str:
        """Navigate to a node by name. Returns full dump of that node's level."""
        nid = self._resolve(name)
        if nid is None:
            candidates = [n for n in self._name_index if name.lower() in n][:6]
            hint = ", ".join(candidates) if candidates else "none"
            return f"Not found: {name!r}. Similar: {hint}"
        self._current = nid
        return self.dump()

    def up(self) -> str:
        """Go one level up."""
        if self._current is None:
            return "Already at MACRO level."
        node = self.g.nodes[self._current]
        if node.parent:
            self._current = node.parent
        else:
            self._current = None
        return self.dump()

    def pwd(self) -> str:
        """Show current position."""
        if self._current is None:
            return "/ (MACRO level)"
        node = self.g.nodes[self._current]
        parts = []
        n = node
        while n:
            parts.append(n.name.split("::")[-1])
            n = self.g.nodes.get(n.parent) if n.parent else None
        return " / ".join(reversed(parts))

    def edges(self, kind: Optional[str] = None) -> str:
        """List edges from current node. kind: CALLS, SEND, DEPENDS_ON, IMPLEMENTS, OWN."""
        if self._current is None:
            return "At MACRO level — cd into a module first."
        target_kind = EdgeKind(kind.upper()) if kind else None
        lines = []
        for edge in self.g.edges.values():
            if edge.src != self._current:
                continue
            if target_kind and edge.kind != target_kind:
                continue
            dst = self.g.nodes.get(edge.dst)
            dst_name = dst.name.split("::")[-1] if dst else edge.dst
            lines.append(f"  {edge.kind.value:12s} -> {dst_name}")
        if not lines:
            return "No outgoing edges from current node."
        cur = self.g.nodes[self._current]
        return f"Edges from [{cur.name.split('::')[-1]}]:\n" + "\n".join(lines)

    def get_code(self, name: str) -> str:
        """Return source code for a MESO (class body) or MICRO (method body)."""
        nid = self._resolve(name)
        if nid is None:
            return f"Node not found: {name!r}"
        node = self.g.nodes[nid]

        if node.level == Level.MACRO:
            return "MACRO nodes have no direct source. cd into it to see classes."

        # extract file path and class from node name: "path/file.hpp::ClassName"
        # MICRO name: "path/file.hpp::ClassName.method_name"
        raw = node.name
        if node.level == Level.MICRO:
            # "path::Class.method"
            path_class, _, method = raw.rpartition(".")
            file_rel, _, class_name = path_class.rpartition("::")
        else:  # MESO
            file_rel, _, class_name = raw.rpartition("::")
            method = None

        src = self._read_file(file_rel)
        if src is None:
            return f"Source file not found: {file_rel}"

        snippet = _extract_node_source(src, class_name, method)
        header = f"// {file_rel}  [{node.level.value}] {class_name}"
        if method:
            header += f".{method}"
        return f"{header}\n{snippet}"

    def relations(self, name: str) -> str:
        """Show all incoming and outgoing edges for any named node."""
        nid = self._resolve(name)
        if nid is None:
            return f"Node not found: {name!r}"
        node = self.g.nodes[nid]
        short = node.name.split("::")[-1]
        lines = [
            f"=== {short} [{node.level.value}] ===",
            f"  {node.intent}",
            "",
        ]
        outgoing, incoming = [], []
        for edge in self.g.edges.values():
            if edge.src == nid:
                dst = self.g.nodes.get(edge.dst)
                if dst:
                    outgoing.append(f"  {edge.kind.value:12s} -> [{dst.level.value:5s}] {dst.name.split('::')[-1]:30s}  {dst.intent[:50]}")
            elif edge.dst == nid:
                src = self.g.nodes.get(edge.src)
                if src:
                    incoming.append(f"  {edge.kind.value:12s} <- [{src.level.value:5s}] {src.name.split('::')[-1]:30s}  {src.intent[:50]}")
        if outgoing:
            lines.append("OUT:")
            lines.extend(outgoing)
        if incoming:
            if outgoing:
                lines.append("")
            lines.append("IN:")
            lines.extend(incoming)
        if not outgoing and not incoming:
            lines.append("No edges.")
        return "\n".join(lines)

    def list_file(self, rel_path: str) -> str:
        """List all classes and methods in a file with line ranges. No source copy-paste."""
        norm = rel_path.replace("\\", "/").strip()
        matches = [n for n in self.g.nodes.values()
                   if n.level == Level.MESO and
                   (n.name.startswith(norm + "::") or
                    n.name.split("::")[0].endswith("/" + norm) or
                    n.name.split("::")[0].endswith(norm))]
        if not matches:
            matches = [n for n in self.g.nodes.values()
                       if n.level == Level.MESO and norm in n.name.split("::")[0]]
        if not matches:
            return f"No classes found for: {rel_path!r}. Use CD to browse modules."

        file_path = matches[0].name.split("::")[0]
        lines_out = [f"=== {file_path} ==="]
        for meso in sorted(matches, key=lambda n: getattr(n.payload, "line_start", 0)):
            class_name = meso.name.split("::")[-1]
            ls = getattr(meso.payload, "line_start", 0)
            le = getattr(meso.payload, "line_end", 0)
            rng = f"  [{ls}-{le}]" if ls else ""
            lines_out.append(f"  class {class_name}{rng}")
            for cid in meso.children:
                child = self.g.nodes.get(cid)
                if not child or child.level != Level.MICRO:
                    continue
                method = child.name.split(".")[-1]
                mls = getattr(child.payload, "line_start", 0)
                mle = getattr(child.payload, "line_end", 0)
                mrng = f"  [{mls}-{mle}]" if mls else ""
                lines_out.append(f"    {method}(){mrng}")
        return "\n".join(lines_out)

    def find(self, keyword: str) -> str:
        """Search nodes by name/intent containing keyword."""
        kw = keyword.lower()
        results = []
        for node in self.g.nodes.values():
            if kw in node.name.lower() or kw in node.intent.lower():
                short = node.name.split("::")[-1]
                results.append(f"  [{node.level.value:5s}] {short:40s} - {node.intent[:60]}")
        if not results:
            return f"No nodes matching {keyword!r}"
        return f"Found {len(results)} node(s):\n" + "\n".join(results[:30])

    # ── internals ────────────────────────────────────────────────────────────

    def _resolve(self, name: str) -> Optional[str]:
        lower = name.lower()
        # exact match
        if lower in self._name_index:
            return self._name_index[lower]
        # prefer children of current node
        if self._current:
            cur = self.g.nodes[self._current]
            for cid in cur.children:
                child = self.g.nodes.get(cid)
                if child and lower in child.name.lower().split(".")[-1]:
                    return cid
        # partial match across all nodes
        for k, v in self._name_index.items():
            if lower in k:
                return v
        return None

    def _read_file(self, rel_path: str) -> Optional[str]:
        if rel_path not in self._source_cache:
            p = self.root / rel_path
            if not p.exists():
                # try case-insensitive glob
                matches = list(self.root.rglob(p.name))
                if matches:
                    p = matches[0]
                else:
                    return None
            self._source_cache[rel_path] = p.read_text(encoding="utf-8", errors="replace")
        return self._source_cache[rel_path]

    def _dump_macros(self) -> str:
        macros = [n for n in self.g.nodes.values() if n.level == Level.MACRO]
        macros.sort(key=lambda n: n.name)
        lines = ["=== MACRO LEVEL ==="]
        for m in macros:
            child_count = len(m.children)
            lines.append(f"  {m.name:20s}  {m.intent[:55]:55s}  [{child_count} classes]")
        return "\n".join(lines)

    def _dump_meso_level(self, macro_node: Node) -> str:
        import re as _re
        lines = [f"=== {macro_node.name.upper()} ===", f"  {macro_node.intent}", ""]
        for cid in macro_node.children:
            child = self.g.nodes.get(cid)
            if not child or child.level != Level.MESO:
                continue
            class_name = child.name.split("::")[-1]
            behaviors = getattr(child.payload, "behaviors", []) if child.payload else []
            blist = ", ".join(behaviors[:8])
            if len(behaviors) > 8:
                blist += f", +{len(behaviors)-8}"
            # strip generic "Class X" / "Struct X" intents — just label the kind
            intent = child.intent[:45]
            if _re.match(r'^(?:class|struct)\s+\S+$', intent, _re.IGNORECASE):
                intent = "struct" if intent.lower().startswith("struct") else "class"
            method_part = f"  (methods: {blist})" if blist else ""
            lines.append(f"  {class_name} — {intent}{method_part}")
        return "\n".join(lines)

    def _dump_micro_level(self, meso_node: Node) -> str:
        class_name = meso_node.name.split("::")[-1]
        lines = [f"=== {class_name} ===", f"  {meso_node.intent}", ""]
        for cid in meso_node.children:
            child = self.g.nodes.get(cid)
            if not child or child.level != Level.MICRO:
                continue
            method = child.name.split(".")[-1]
            inputs = getattr(child.payload, "inputs", []) if child.payload else []
            sig = f"({', '.join(inputs)})" if inputs else "()"
            # skip intent if it just repeats the method name
            intent = child.intent
            intent_str = f" — {intent}" if intent.lower() != method.lower() else ""
            lines.append(f"  {method}{sig}{intent_str}")
        return "\n".join(lines)
