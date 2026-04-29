from __future__ import annotations
import ast
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional

from gkg import Graph, Level, Status, Phase, EdgeKind, Ownership
from ollama_client import OllamaClient

try:
    from tree_sitter import Language, Parser as TSParser
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False

EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs":    "rust",
    ".cs":    "csharp",
    ".java":  "java",
    ".rb":    "ruby",
    ".swift": "swift",
    ".kt":    "kotlin",
    ".kts":   "kotlin",
    ".php":   "php",
    ".cpp":   "cpp",
    ".cc":    "cpp",
    ".cxx":   "cpp",
    ".c":     "c",
    ".h":     "c",
    ".hpp":   "cpp",
    ".m":     "objc",
    ".scala": "scala",
    ".ex":    "elixir",
    ".exs":   "elixir",
    ".hs":    "haskell",
    ".lua":   "lua",
    ".r":     "r",
    ".R":     "r",
}

IGNORE_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".mypy_cache",
})

# local #include regex — stdlib <...> excluded on purpose
_INCLUDE_RE = re.compile(r'#include\s+"([^"]+)"')
# class Foo : public Bar  (C++/Java/C# inheritance)
_INHERIT_RE = re.compile(
    r'(?:class|struct|interface)\s+(\w+)\b[^{;]*?'
    r'(?:extends|implements|:|:)\s*'
    r'(?:public\s+|protected\s+|private\s+)*(\w+)',
    re.MULTILINE,
)
# Python import: from .module import X  /  import module.X
_PY_IMPORT_RE = re.compile(
    r'^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))',
    re.MULTILINE,
)


# ── intermediate structures ──────────────────────────────────

@dataclass
class FuncInfo:
    name: str
    args: list[str]
    docstring: str
    calls: list[str] = field(default_factory=list)
    sends: list[str] = field(default_factory=list)  # callees whose return value is consumed
    line_start: int = 0
    line_end: int = 0


@dataclass
class ClassInfo:
    name: str
    bases: list[str] = field(default_factory=list)   # parent class names
    methods: list[FuncInfo] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0


@dataclass
class FileInfo:
    rel_path: str           # normalized forward-slash path
    language: str
    src: str                # raw source (for include extraction)
    classes: list[ClassInfo] = field(default_factory=list)
    module_funcs: list[FuncInfo] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)  # resolved rel_paths


# ── Python extraction ────────────────────────────────────────

def _extract_func(node: ast.FunctionDef | ast.AsyncFunctionDef) -> FuncInfo:
    args = [a.arg for a in node.args.args if a.arg != "self"]
    doc = (ast.get_docstring(node) or "")[:200]
    calls: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)
    return FuncInfo(name=node.name, args=args, docstring=doc, calls=list(set(calls)),
                    line_start=node.lineno, line_end=getattr(node, "end_lineno", 0))


def _extract_python(src: str) -> tuple[list[ClassInfo], list[FuncInfo]]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], []
    classes: list[ClassInfo] = []
    module_funcs: list[FuncInfo] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(b.attr)
            ci = ClassInfo(name=node.name, bases=bases,
                           line_start=node.lineno,
                           line_end=getattr(node, "end_lineno", 0))
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ci.methods.append(_extract_func(item))
            classes.append(ci)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_funcs.append(_extract_func(node))
    return classes, module_funcs


# ── tree-sitter extraction ───────────────────────────────────

def _load_ts_lang(lang_name: str) -> Optional["Language"]:
    if not _TS_AVAILABLE:
        return None
    try:
        if lang_name == "javascript":
            import tree_sitter_javascript as m
            return Language(m.language())
        elif lang_name == "typescript":
            import tree_sitter_typescript as m
            return Language(m.language_typescript())
        elif lang_name == "go":
            import tree_sitter_go as m
            return Language(m.language())
        elif lang_name in ("cpp", "c"):
            import tree_sitter_cpp as m
            return Language(m.language())
    except (ImportError, AttributeError):
        pass
    return None


def _ts_name(node) -> str:
    for c in node.children:
        if c.type in ("identifier", "property_identifier", "field_identifier", "type_identifier"):
            return c.text.decode(errors="replace")
    return ""


def _collect_methods(node, ci: ClassInfo) -> None:
    if node.type in ("method_definition", "function_declaration", "function_definition",
                     "method_declaration"):
        name = _ts_name(node)
        if name:
            ci.methods.append(FuncInfo(name=name, args=[], docstring=""))
    for child in node.children:
        _collect_methods(child, ci)


def _extract_treesitter(src: str, lang_name: str) -> tuple[list[ClassInfo], list[FuncInfo]]:
    lang = _load_ts_lang(lang_name)
    if lang is None:
        return [], []
    parser = TSParser(lang)
    tree = parser.parse(src.encode(errors="replace"))
    classes: list[ClassInfo] = []

    if lang_name in ("cpp", "c"):
        return _extract_ts_cpp(tree.root_node, src), []

    module_funcs: list[FuncInfo] = []

    def _walk(node, depth: int = 0) -> None:
        if node.type in ("class_declaration", "class_definition", "type_declaration"):
            name = _ts_name(node)
            if name:
                ci = ClassInfo(name=name)
                for child in node.children:
                    _collect_methods(child, ci)
                classes.append(ci)
            return
        if node.type in ("function_declaration", "function_definition") and depth == 0:
            name = _ts_name(node)
            if name:
                module_funcs.append(FuncInfo(name=name, args=[], docstring=""))
        for child in node.children:
            _walk(child, depth)

    _walk(tree.root_node)
    return classes, module_funcs


def _ts_cpp_class_name(node) -> str:
    """Extract class/struct name from a tree-sitter C++ class_specifier node."""
    for c in node.children:
        if c.type == "type_identifier":
            return c.text.decode(errors="replace")
    return ""


def _ts_cpp_func_name(node) -> str:
    """Recursively find method name inside a declarator subtree.

    Handles:
      declaration      → function_declarator → identifier          (ctor)
      declaration      → function_declarator → destructor_name     (dtor)
      field_declaration→ function_declarator → field_identifier    (method)
      field_declaration→ reference_declarator→ function_declarator → operator_name
      function_definition (inline) — same as above
    """
    for c in node.children:
        if c.type in ("function_declarator", "reference_declarator",
                      "pointer_declarator", "abstract_function_declarator"):
            name = _ts_cpp_func_name(c)
            if name:
                return name
        if c.type in ("identifier", "field_identifier"):
            return c.text.decode(errors="replace")
        if c.type == "destructor_name":
            return c.text.decode(errors="replace")
        if c.type == "operator_name":
            return c.text.decode(errors="replace")
    return ""


def _has_func_declarator(node) -> bool:
    """True if node or any direct/1-deep child is a function_declarator."""
    for c in node.children:
        if c.type == "function_declarator":
            return True
        for cc in c.children:
            if cc.type == "function_declarator":
                return True
    return False


_SEND_PARENT_TYPES = frozenset({
    "return_statement", "assignment_expression", "init_declarator",
    "binary_expression", "condition_clause", "argument_list",
    "parenthesized_expression",
})


def _ts_cpp_call_sites(body_node) -> tuple[list[str], list[str]]:
    """Walk a function body node, return (calls, sends).

    calls: every callee name invoked
    sends: callee names whose return value is consumed (assigned, returned, passed as arg)
    """
    calls: list[str] = []
    sends: list[str] = []

    def _callee_name(call_node) -> str:
        fn = next(
            (c for c in call_node.children
             if c.is_named and c.type != "argument_list"),
            None,
        )
        if fn is None:
            return ""
        if fn.type == "identifier":
            return fn.text.decode(errors="replace")
        if fn.type == "field_expression":
            for c in fn.children:
                if c.type == "field_identifier":
                    return c.text.decode(errors="replace")
        if fn.type == "qualified_identifier":
            ids = [c for c in fn.children if c.type in ("identifier", "destructor_name")]
            return ids[-1].text.decode(errors="replace") if ids else ""
        return ""

    def _walk(n) -> None:
        if n.type == "call_expression":
            name = _callee_name(n)
            if name and name not in _CPP_KEYWORDS:
                calls.append(name)
                p = n.parent
                if p is not None and p.type in _SEND_PARENT_TYPES:
                    sends.append(name)
        for child in n.children:
            _walk(child)

    _walk(body_node)
    return calls, sends


def _ts_cpp_methods(class_body_node) -> list[FuncInfo]:
    """Extract methods from a C++ field_declaration_list node.
    For inline function_definition nodes, also extract call sites and send targets.
    """
    methods: list[FuncInfo] = []
    seen: set[str] = set()

    def _try_add(node) -> None:
        if not _has_func_declarator(node):
            return
        name = _ts_cpp_func_name(node)
        if not name or name in seen:
            return
        seen.add(name)
        calls: list[str] = []
        sends: list[str] = []
        if node.type == "function_definition":
            body = next((c for c in node.children if c.type == "compound_statement"), None)
            if body:
                calls, sends = _ts_cpp_call_sites(body)
        ls = node.start_point[0] + 1
        le = node.end_point[0] + 1
        methods.append(FuncInfo(name=name, args=[], docstring="", calls=calls, sends=sends,
                                line_start=ls, line_end=le))

    for child in class_body_node.children:
        if child.type in ("function_definition", "declaration", "field_declaration"):
            _try_add(child)
        elif child.type == "template_declaration":
            for tc in child.children:
                if tc.type in ("function_definition", "declaration", "field_declaration"):
                    _try_add(tc)

    return methods


def _extract_ts_cpp(root, src: str) -> list[ClassInfo]:
    """Walk tree-sitter C++ AST, extract classes/structs with their methods."""
    classes: list[ClassInfo] = []
    seen: set[str] = set()

    def _walk(node) -> None:
        if node.type in ("class_specifier", "struct_specifier"):
            name = _ts_cpp_class_name(node)
            if name and name not in seen:
                seen.add(name)
                methods: list[FuncInfo] = []
                for c in node.children:
                    if c.type == "field_declaration_list":
                        methods = _ts_cpp_methods(c)
                # infer inheritance from base_class_clause
                bases: list[str] = []
                for c in node.children:
                    if c.type == "base_class_clause":
                        for bc in c.children:
                            if bc.type == "type_identifier":
                                bases.append(bc.text.decode(errors="replace"))
                ls = node.start_point[0] + 1
                le = node.end_point[0] + 1
                classes.append(ClassInfo(name=name, bases=bases, methods=methods,
                                         line_start=ls, line_end=le))
        # don't skip into class bodies handled above — recurse everything else
        for child in node.children:
            _walk(child)

    _walk(root)
    return classes


# ── generic regex extractor ──────────────────────────────────

_GENERIC_CLASS_RE = re.compile(
    r'^\s*(?:public\s+|private\s+|protected\s+|abstract\s+|sealed\s+|open\s+|data\s+|'
    r'final\s+|static\s+)*'
    r'(?:class|struct|interface|trait|enum|impl|object|record)\s+'
    r'(?:class\s+)?'           # consume "class" in "enum class Name" (C++ scoped enum)
    r'([A-Za-z_][A-Za-z0-9_]*)',
    re.MULTILINE,
)
_GENERIC_FUNC_RE = re.compile(
    r'^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+|override\s+|'
    r'virtual\s+|abstract\s+|inline\s+|unsafe\s+|extern\s+)?'
    r'(?:fun|fn|def|func|function|sub|procedure)\s+'
    r'([A-Za-z_][A-Za-z0-9_]*)\s*\(',
    re.MULTILINE,
)
_TYPED_FUNC_RE = re.compile(
    r'^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+|override\s+|'
    r'virtual\s+|abstract\s+|readonly\s+|unsafe\s+|inline\s+|constexpr\s+)*'
    r'(?:void|int|long|float|double|bool|string|String|char|byte|object|Object|'
    r'auto|var|let|const|[A-Z][A-Za-z0-9_<>\[\]*&]*)\s+'
    r'([a-z_][A-Za-z0-9_]*)\s*\(',
    re.MULTILINE,
)
_CPP_BASES_RE = re.compile(
    r'(?:class|struct)\s+(\w+)\s*:[^{;]+\{',
    re.MULTILINE,
)
_CALL_SITE_RE = re.compile(r'\b([a-z_][A-Za-z0-9_]*)\s*\(')
_CPP_KEYWORDS = frozenset({
    'if', 'else', 'while', 'for', 'switch', 'case', 'return', 'break',
    'continue', 'do', 'goto', 'sizeof', 'static', 'const', 'auto',
    'new', 'delete', 'throw', 'catch', 'try', 'namespace', 'using',
    'typename', 'template', 'decltype', 'nullptr', 'true', 'false',
    'void', 'int', 'long', 'float', 'double', 'bool', 'char', 'short',
    'unsigned', 'signed', 'inline', 'explicit', 'operator',
    'assert', 'printf', 'fprintf', 'sprintf', 'malloc', 'free',
})


def _extract_cpp_bases(src: str) -> list[tuple[str, list[str]]]:
    results = []
    for m in _CPP_BASES_RE.finditer(src):
        class_name = m.group(1)
        inherited = re.findall(r'(?:public|protected|private)\s+(\w+)', m.group(0))
        if inherited:
            results.append((class_name, inherited))
    return results


def _body_range(src: str, pos: int) -> tuple[int, int]:
    """Return (open_brace_pos, close_brace_pos+1). Returns (-1,-1) if ';' hit first."""
    i = pos
    while i < len(src):
        c = src[i]
        if c == ';':
            return -1, -1
        if c == '{':
            break
        i += 1
    if i >= len(src):
        return -1, -1
    depth = 0
    start = i
    while i < len(src):
        if src[i] == '{':
            depth += 1
        elif src[i] == '}':
            depth -= 1
            if depth == 0:
                return start, i + 1
        i += 1
    return -1, -1


def _func_body(src: str, pos: int) -> str:
    s, e = _body_range(src, pos)
    return src[s + 1: e - 1] if s != -1 else ""


def _call_names(body: str) -> list[str]:
    return list({m.group(1) for m in _CALL_SITE_RE.finditer(body)
                 if m.group(1) not in _CPP_KEYWORDS})


def _depth0_text(text: str) -> str:
    """Strip all nested {…} content — return only depth-0 characters."""
    out: list[str] = []
    depth = 0
    for c in text:
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        elif depth == 0:
            out.append(c)
    return ''.join(out)


# Keywords that can appear before '(' but are NOT method names
_METHOD_SKIP = frozenset({
    'if', 'else', 'while', 'for', 'switch', 'case', 'do', 'return',
    'catch', 'throw', 'new', 'delete', 'sizeof', 'alignof', 'decltype',
    'static_cast', 'dynamic_cast', 'reinterpret_cast', 'const_cast',
    'assert', 'printf', 'fprintf', 'sprintf',
    'public', 'private', 'protected', 'explicit', 'virtual', 'inline',
    'constexpr', 'static', 'const', 'volatile', 'override', 'final',
    'noexcept', 'typename', 'template', 'namespace', 'using', 'operator',
    'void', 'int', 'long', 'float', 'double', 'bool', 'char', 'short',
    'unsigned', 'signed', 'auto', 'nullptr', 'true', 'false',
})
_METHOD_DECL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*\(')


def _extract_class_methods(class_body: str) -> list[FuncInfo]:
    """Extract methods at depth-0 of a class body using broad identifier( matching."""
    flat = _depth0_text(class_body)
    methods: list[FuncInfo] = []
    seen: set[str] = set()
    for m in _METHOD_DECL_RE.finditer(flat):
        name = m.group(1)
        if name in _METHOD_SKIP or name in seen:
            continue
        seen.add(name)
        # find call sites from this method's inline body in the original text
        calls: list[str] = []
        try:
            for hit in re.finditer(rf'\b{re.escape(name)}\s*\(', class_body):
                body = _func_body(class_body, hit.start())
                if body:
                    calls = _call_names(body)
                    break
        except re.error:
            pass
        methods.append(FuncInfo(name=name, args=[], docstring="", calls=calls))
    return methods


def _extract_generic(src: str) -> tuple[list[ClassInfo], list[FuncInfo]]:
    """Extract classes (with methods) from source. Free functions intentionally ignored —
    only OOP structures (class/struct) produce GKG nodes."""
    classes: list[ClassInfo] = []
    seen_classes: set[str] = set()
    cpp_bases = {name: bases for name, bases in _extract_cpp_bases(src)}

    for m in _GENERIC_CLASS_RE.finditer(src):
        name = m.group(1)
        if name in seen_classes or name in _METHOD_SKIP:
            continue
        seen_classes.add(name)
        s, e = _body_range(src, m.end())
        if s != -1:
            methods = _extract_class_methods(src[s + 1: e - 1])
        else:
            methods = []
        classes.append(ClassInfo(name=name, bases=cpp_bases.get(name, []),
                                 methods=methods))

    # Free functions not returned — they don't become GKG nodes.
    return classes, []


_TS_LANGS = {"python", "javascript", "typescript", "go", "cpp", "c"}


# ── include resolution ───────────────────────────────────────

def _resolve_includes(src: str, file_rel: str, known_paths: set[str]) -> list[str]:
    """Parse #include "..." and resolve to known file rel-paths (forward slashes)."""
    file_dir = str(PurePosixPath(file_rel).parent)
    resolved: list[str] = []
    for m in _INCLUDE_RE.finditer(src):
        raw = m.group(1).replace("\\", "/")
        norm = os.path.normpath(os.path.join(file_dir, raw)).replace("\\", "/")
        if norm in known_paths:
            resolved.append(norm)
        else:
            # fallback: -I search-path style — match any known path ending with raw
            suffix = "/" + raw
            matches = [p for p in known_paths if p.endswith(suffix) or p == raw]
            if len(matches) == 1:
                resolved.append(matches[0])
    return resolved


# ── directory scan ────────────────────────────────────────────

def _scan(path: str) -> list[FileInfo]:
    root = Path(path)
    files: list[FileInfo] = []
    ext_counter: Counter = Counter()

    for fpath in sorted(root.rglob("*")):
        if not fpath.is_file():
            continue
        if any(part in IGNORE_DIRS for part in fpath.parts):
            continue
        ext = fpath.suffix.lower()
        lang = EXT_TO_LANG.get(ext)
        if not lang:
            if ext:
                ext_counter[ext] += 1
            continue
        try:
            src = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        if lang == "python":
            classes, funcs = _extract_python(src)
        elif lang in _TS_LANGS:
            classes, funcs = _extract_treesitter(src, lang)
            if not classes and not funcs and lang not in ("cpp", "c"):
                classes, funcs = _extract_generic(src)
        else:
            classes, funcs = _extract_generic(src)

        if classes or funcs:
            rel = str(fpath.relative_to(root)).replace("\\", "/")
            files.append(FileInfo(
                rel_path=rel,
                language=lang,
                src=src,
                classes=classes,
                module_funcs=funcs,
            ))

    if not files and ext_counter:
        top = ext_counter.most_common(8)
        hint = ", ".join(f"{e}({n})" for e, n in top)
        raise ValueError(
            f"no supported source files found in {path!r}.\n"
            f"Extensions present (unsupported): {hint}\n"
            f"Add them to EXT_TO_LANG in ast_mapper.py."
        )

    # resolve includes now that we know all paths
    known = {fi.rel_path for fi in files}
    for fi in files:
        fi.includes = _resolve_includes(fi.src, fi.rel_path, known)

    return files


# ── AI clustering + intent labeling ──────────────────────────

_CLUSTER_SYS = """\
You fix and verify module groupings for a codebase. Classes are pre-grouped by folder name.
Your job: rename vague modules, merge tiny ones, and move misplaced classes.

Input JSON:
{
  "modules": [{"name": str, "classes": [{"name": str, "methods": [str]}]}],
  "sigs":    [{"name": str, "args": [str], "docstring": str}]
}

Output ONE JSON:
{
  "modules": [
    {
      "name":    str,
      "intent":  str,
      "classes": [class_name_str]
    }
  ],
  "intents": [{"name": str, "intent": str}]
}

Naming rules:
- Keep existing module name if it is already specific and descriptive.
- Rename only if the name is too generic: "utils", "common", "helpers", "misc",
  "shared", "base", "core" (when ambiguous), "root", "src", "lib", "include".
- Good module names describe what the code DOES or IS, in the project's own vocabulary.
  Pattern: [domain_verb_or_noun] — short, lowercase, 1-2 words.
  Examples by domain (do NOT copy these; derive names from the actual class names):
    graphics engine  → "rendering", "geometry", "scene"
    web backend      → "auth", "routing", "persistence"
    data pipeline    → "ingestion", "transformation", "export"
    game engine      → "physics", "audio", "input"
- Merge modules with only 1 class into a semantically related larger module when obvious.
- Every class must appear in exactly one output module.
- intents: one sentence per function/method, same name key as input sigs.
- No extra keys.
"""

# Folder names that are semantically meaningful — stop here when scanning path
_KNOWN_SEMANTIC = frozenset({
    # architecture / system
    'core', 'base', 'common', 'foundation', 'kernel',
    # network
    'net', 'network', 'networking', 'socket', 'http', 'rpc', 'grpc', 'api',
    'client', 'server', 'proxy', 'protocol', 'transport',
    # storage / IO
    'io', 'fs', 'filesystem', 'storage', 'cache', 'db', 'database',
    'persist', 'persistence', 'repository', 'store',
    # graphics / rendering
    'render', 'rendering', 'renderer', 'graphics', 'draw', 'drawing',
    'visual', 'display', 'image', 'canvas', 'texture', 'shader',
    # GPU
    'gpu', 'cuda', 'opencl', 'vulkan', 'opengl', 'metal', 'directx',
    # UI
    'ui', 'gui', 'widget', 'widgets', 'view', 'views', 'window', 'layout',
    # media
    'audio', 'sound', 'video', 'media', 'codec', 'stream',
    # simulation
    'physics', 'collision', 'sim', 'simulation', 'dynamics',
    # game / scene
    'animation', 'anim', 'scene', 'entity', 'world', 'actor', 'component',
    # math
    'math', 'geometry', 'algebra', 'vector', 'matrix', 'numeric',
    # algorithms
    'algorithm', 'algorithms', 'algo', 'search', 'sort', 'graph',
    # security
    'crypto', 'security', 'auth', 'ssl', 'tls', 'hash', 'cipher',
    # memory
    'memory', 'alloc', 'allocator', 'pool', 'arena', 'heap',
    # concurrency
    'thread', 'threading', 'async', 'sync', 'concurrency', 'parallel',
    'executor', 'scheduler', 'task', 'fiber',
    # diagnostics
    'log', 'logging', 'debug', 'trace', 'profiler', 'telemetry',
    # config
    'config', 'settings', 'options', 'params', 'param', 'configuration',
    # data / model
    'data', 'model', 'models', 'schema', 'serialize', 'serialization',
    'format', 'codec', 'encode', 'decode', 'parse', 'parsing',
    # compiler / language tools
    'parser', 'lexer', 'compiler', 'codegen', 'ir', 'ast', 'bytecode',
    # plugin / extension
    'plugin', 'plugins', 'extension', 'extensions', 'module', 'modules',
    # platform / OS
    'platform', 'os', 'sys', 'system', 'native', 'posix', 'win32',
    # events / messaging
    'event', 'events', 'signal', 'signals', 'message', 'queue', 'bus',
    'dispatch', 'pubsub', 'notify',
    # error handling
    'error', 'errors', 'exception', 'diagnostic', 'result',
    # utility (intentionally kept — we know what to do with these)
    'util', 'utils', 'utility', 'utilities', 'helper', 'helpers', 'tools',
    # pipeline / transform
    'output', 'input', 'transform', 'pipeline', 'filter',
    # plotting / charting
    'plot', 'chart', 'axis', 'tick', 'figure', 'plot2d', 'plot3d',
    # testing
    'test', 'tests', 'bench', 'benchmark', 'stress', 'perf',
    # examples / demos
    'examples', 'demo', 'sample', 'samples',
})

# Structural-only dirs — skip when looking for module name
_STRUCTURAL_DIRS = frozenset({
    'src', 'source', 'sources', 'lib', 'libs', 'include', 'includes',
    'build', 'dist', 'out', 'bin', 'obj', 'cmake', 'scripts', 'third_party',
    'external', 'vendor', 'deps', 'dependencies',
})

# Module names too generic to use as-is — trigger LLM refinement
_VAGUE_NAMES = frozenset({
    "utils", "util", "common", "helpers", "helper", "misc", "shared",
    "base", "root", "src", "lib", "include", "source", "main", "core",
})


def _needs_llm_refinement(class_to_module: dict[str, str]) -> bool:
    """True if any module name is vague or any module has only 1 class (merge candidate)."""
    if any(m in _VAGUE_NAMES for m in class_to_module.values()):
        return True
    counts: Counter = Counter(class_to_module.values())
    single_class_modules = sum(1 for c in counts.values() if c == 1)
    return single_class_modules > len(counts) // 2   # majority are singletons


def _ai_cluster_and_label(
    files: list[FileInfo],
    client: OllamaClient,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """
    Folder-first: _dir_cluster gives initial groupings from directory names.
    LLM called only when module names are vague or too many singleton modules exist.
    Returns (class_to_module, stray_to_module, module_intents, name_intents).
    """
    all_class_names = {ci.name for fi in files for ci in fi.classes}
    all_stray_paths  = {fi.rel_path for fi in files if not fi.classes}

    # ── step 1: folder-based grouping (always) ───────────────
    class_to_module, stray_to_module = _dir_cluster(files)

    # ── step 2: build sigs for intent labeling ───────────────
    sigs: list[dict] = []
    for fi in files:
        for ci in fi.classes:
            for m in ci.methods:
                sigs.append({"name": f"{ci.name}.{m.name}", "args": m.args,
                             "docstring": m.docstring[:150]})

    # ── step 3: LLM refinement only if needed ────────────────
    module_intents: dict[str, str] = {}
    name_intents:   dict[str, str] = {}

    if _needs_llm_refinement(class_to_module):
        # build module→classes view for LLM input
        mod_classes: dict[str, list[dict]] = defaultdict(list)
        for fi in files:
            for ci in fi.classes:
                mod = class_to_module.get(ci.name, "misc")
                mod_classes[mod].append({"name": ci.name,
                                          "methods": [m.name for m in ci.methods]})
        modules_input = [{"name": k, "classes": v} for k, v in sorted(mod_classes.items())]

        sigs_chunk = sigs[:60]
        payload = json.dumps({"modules": modules_input, "sigs": sigs_chunk})

        try:
            raw = client.complete_json(payload, system=_CLUSTER_SYS,
                                       max_tokens=6000, label="cluster")
            for mod in raw.get("modules", []):
                mod_name   = mod.get("name", "misc")
                mod_intent = mod.get("intent", f"Module {mod_name}")
                module_intents[mod_name] = mod_intent
                for cls in mod.get("classes", []):
                    if cls in all_class_names:
                        class_to_module[cls] = mod_name

            name_intents = {item["name"]: item["intent"]
                            for item in raw.get("intents", [])
                            if "name" in item and "intent" in item}

            # label remaining sigs
            for i in range(len(sigs_chunk), len(sigs), 40):
                chunk = sigs[i: i + 40]
                try:
                    r2 = client.complete_json(
                        json.dumps(chunk),
                        system=("Label code functions with one-sentence intents.\n"
                                "Input: JSON array of {name, args, docstring}.\n"
                                "Output: JSON array of {name, intent}."),
                        max_tokens=4000, label="intent",
                    )
                    if isinstance(r2, list):
                        for item in r2:
                            if "name" in item and "intent" in item:
                                name_intents[item["name"]] = item["intent"]
                except Exception:
                    pass
        except Exception:
            pass  # keep folder-based grouping as-is
    else:
        # folders are clean — only call LLM for intent labeling
        sigs_chunk = sigs[:60]
        for i in range(0, len(sigs), 40):
            chunk = sigs[i: i + 40]
            try:
                r2 = client.complete_json(
                    json.dumps(chunk),
                    system=("Label code functions with one-sentence intents.\n"
                            "Input: JSON array of {name, args, docstring}.\n"
                            "Output: JSON array of {name, intent}."),
                    max_tokens=4000, label="intent",
                )
                if isinstance(r2, list):
                    for item in r2:
                        if "name" in item and "intent" in item:
                            name_intents[item["name"]] = item["intent"]
            except Exception:
                pass

    # ensure all stray paths assigned
    fb_cls, fb_stray = _dir_cluster(files)
    for fp in all_stray_paths:
        stray_to_module.setdefault(fp, fb_stray.get(fp, "misc"))

    return class_to_module, stray_to_module, module_intents, name_intents


def _dir_cluster(files: list[FileInfo]) -> tuple[dict[str, str], dict[str, str]]:
    """Group classes/files into modules using directory-name anchors.

    Priority:
      1. First path component in _KNOWN_SEMANTIC  → use it as module name
      2. First non-structural, non-root component  → use it
      3. Immediate parent dir (parts[-2])          → fallback
      4. File stem                                 → last resort (no parent dir)
    """
    def _key(rel_path: str) -> str:
        parts = PurePosixPath(rel_path).parts[:-1]   # dirs only, no filename
        if not parts:
            return "root"

        # 1. known semantic anchor — scan left-to-right, skip structural wrappers
        for part in parts:
            p = part.lower()
            if p in _STRUCTURAL_DIRS:
                continue
            if p in _KNOWN_SEMANTIC:
                return p

        # 2. first non-structural component
        for part in parts:
            p = part.lower()
            if p not in _STRUCTURAL_DIRS:
                return p

        # 3. immediate parent
        return parts[-1].lower()

    class_to_module: dict[str, str] = {}
    stray_to_module: dict[str, str] = {}
    for fi in files:
        k = _key(fi.rel_path)
        for ci in fi.classes:
            class_to_module[ci.name] = k
        if not fi.classes:
            stray_to_module[fi.rel_path] = k
    return class_to_module, stray_to_module


# ── graph builder ─────────────────────────────────────────────

def map_project(path: str, client: Optional[OllamaClient] = None) -> Graph:
    """
    Map a codebase to a GKG.
    - Structure (nodes, edges): deterministic AST/regex.
    - MACRO clustering + intent labels: AI (optional; falls back to dir grouping).
    MACRO = semantic module (cluster of classes), MESO = class, MICRO = method/function.
    """
    files = _scan(path)
    if not files:
        raise ValueError(f"no supported source files found in {path!r}")

    if client:
        class_to_module, stray_to_module, module_intents, name_intents = \
            _ai_cluster_and_label(files, client)
    else:
        class_to_module, stray_to_module = _dir_cluster(files)
        module_intents, name_intents = {}, {}

    def ni(key: str, fallback: str) -> str:
        return name_intents.get(key, fallback)

    # ── determine module names from classes only (stray files produce no nodes) ─
    module_langs: dict[str, Counter] = defaultdict(Counter)
    for fi in files:
        for ci in fi.classes:
            mod = class_to_module.get(ci.name, "misc")
            module_langs[mod][fi.language] += 1
    all_modules: set[str] = set(module_langs.keys())

    g = Graph()
    name_to_id: dict[str, str] = {}

    # ── MACROs (one per semantic module) ────────────────────
    macro_ids: list[str] = []
    for mod_name in sorted(all_modules):
        langs = module_langs[mod_name]
        lang  = langs.most_common(1)[0][0] if langs else "unknown"
        mid = g.add_macro(
            name=mod_name,
            intent=module_intents.get(mod_name, f"Module {mod_name}"),
            language=lang,
        )
        macro_ids.append(mid)
        name_to_id[mod_name] = mid

    if macro_ids:
        g.promote_group(macro_ids, Status.LAYER_STABLE)
        g.advance_phase(Phase.MACRO_STABLE)

    # ── MESOs (one per class only — no synthetic free-function MESOs) ──
    meso_ids: list[str] = []

    for fi in files:
        for ci in fi.classes:
            mod_name = class_to_module.get(ci.name, "misc")
            macro_id = name_to_id.get(mod_name)
            if not macro_id:
                continue
            node_name = f"{fi.rel_path}::{ci.name}"
            cid = g.add_meso(
                parent=macro_id,
                name=node_name,
                intent=ni(ci.name, f"Class {ci.name}"),
                behaviors=[m.name for m in ci.methods],
                line_start=ci.line_start,
                line_end=ci.line_end,
            )
            meso_ids.append(cid)
            name_to_id[node_name] = cid
            name_to_id.setdefault(ci.name, cid)

    if meso_ids:
        g.promote_group(meso_ids, Status.LAYER_STABLE)
        g.advance_phase(Phase.MESO_STABLE)

    # ── MICROs (class methods only) ──────────────────────────
    micro_ids: list[str] = []

    for fi in files:
        for ci in fi.classes:
            class_id = name_to_id.get(f"{fi.rel_path}::{ci.name}")
            if not class_id:
                continue
            seen_methods: set[str] = set()
            for m in ci.methods:
                if m.name in seen_methods:
                    continue
                seen_methods.add(m.name)
                qname = f"{fi.rel_path}::{ci.name}.{m.name}"
                fid = g.add_micro(
                    parent=class_id,
                    name=qname,
                    intent=ni(f"{ci.name}.{m.name}", m.name),
                    inputs=m.args,
                    line_start=m.line_start,
                    line_end=m.line_end,
                )
                micro_ids.append(fid)
                name_to_id[qname] = fid

    if micro_ids:
        g.promote_group(micro_ids, Status.LAYER_STABLE)
        g.advance_phase(Phase.MICRO_STABLE)

    # ── file → macro map (needed for DEPENDS_ON) ────────────
    file_to_macro: dict[str, str] = {}
    for fi in files:
        if fi.classes:
            mod = class_to_module.get(fi.classes[0].name, "misc")
        else:
            mod = stray_to_module.get(fi.rel_path, "misc")
        mid = name_to_id.get(mod)
        if mid:
            file_to_macro[fi.rel_path] = mid

    # ── DEPENDS_ON edges (file includes → macro-level) ───────
    macro_dep_seen: set[tuple[str, str]] = set()
    for fi in files:
        src_macro = file_to_macro.get(fi.rel_path)
        if not src_macro:
            continue
        for inc_path in fi.includes:
            dst_macro = file_to_macro.get(inc_path)
            if dst_macro and dst_macro != src_macro:
                key = (src_macro, dst_macro)
                if key not in macro_dep_seen:
                    macro_dep_seen.add(key)
                    try:
                        g.add_edge(src_macro, dst_macro, EdgeKind.DEPENDS_ON)
                    except Exception:
                        pass

    # ── IMPLEMENTS edges (class inheritance) ─────────────────
    for fi in files:
        for ci in fi.classes:
            sub_id = name_to_id.get(f"{fi.rel_path}::{ci.name}")
            if not sub_id:
                continue
            for base_name in ci.bases:
                base_id = name_to_id.get(base_name)
                if base_id and base_id != sub_id:
                    try:
                        g.add_edge(sub_id, base_id, EdgeKind.IMPLEMENTS)
                    except Exception:
                        pass

    # ── CALLS edges (all languages — Python from ast.walk, C++/others from body regex) ─
    # pre-build a name → [node_id] lookup for fast callee resolution
    callee_index: dict[str, list[str]] = defaultdict(list)
    for key, nid in name_to_id.items():
        short = key.split("::")[-1].split(".")[-1]
        callee_index[short].append(nid)

    calls_seen: set[tuple[str, str]] = set()

    for fi in files:
        all_funcs: list[tuple[str, FuncInfo]] = []
        for ci in fi.classes:
            for m in ci.methods:
                all_funcs.append((f"{fi.rel_path}::{ci.name}.{m.name}", m))

        for caller_key, func in all_funcs:
            caller_id = name_to_id.get(caller_key)
            if not caller_id:
                continue
            for callee_name in func.calls:
                for nid in callee_index.get(callee_name, []):
                    if nid == caller_id:
                        continue
                    pair = (caller_id, nid)
                    if pair not in calls_seen:
                        calls_seen.add(pair)
                        try:
                            g.add_edge(caller_id, nid, EdgeKind.CALLS)
                        except Exception:
                            pass

    # ── SEND edges (callee → caller when return value is consumed) ──
    sends_seen: set[tuple[str, str]] = set()

    for fi in files:
        for ci in fi.classes:
            for m in ci.methods:
                caller_key = f"{fi.rel_path}::{ci.name}.{m.name}"
                caller_id = name_to_id.get(caller_key)
                if not caller_id:
                    continue
                for callee_name in m.sends:
                    for nid in callee_index.get(callee_name, []):
                        if nid == caller_id:
                            continue
                        pair = (nid, caller_id)   # callee → caller
                        if pair not in sends_seen:
                            sends_seen.add(pair)
                            try:
                                g.add_edge(nid, caller_id, EdgeKind.SEND)
                            except Exception:
                                pass

    # ── OWN edges (parent → child for every non-root node) ───
    for n in g.nodes.values():
        if n.parent is not None:
            try:
                g.add_edge(n.parent, n.id, EdgeKind.OWN)
            except Exception:
                pass

    return g
