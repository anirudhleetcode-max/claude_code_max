"""Symbol and import extraction.

Python uses :mod:`ast` (regex fallback on syntax errors). JavaScript/TypeScript,
Go, Rust, Java, Kotlin and C# use conservative regexes combined with a brace
scanner that skips strings and comments, so members are only attributed to a
class when they sit directly in its body.
"""

from __future__ import annotations

import ast
import bisect
import posixpath
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

_MAX_SIGNATURE = 200


class Symbol(BaseModel):
    """A named definition in a source file."""

    name: str
    kind: str  # class, function, method, interface, type, struct, enum, const, variable
    line: int
    end_line: int | None = None
    parent: str | None = None
    signature: str = ""


def _clean_sig(text: str) -> str:
    sig = " ".join(text.split())
    if sig.endswith("{"):
        sig = sig[:-1].rstrip()
    return sig[:_MAX_SIGNATURE]


# --------------------------------------------------------------------------- python


_PY_CONST = re.compile(r"^[A-Z][A-Z0-9_]*$")
_PARSE_CACHE = threading.local()


def _parse_python(text: str) -> ast.Module | None:
    """``ast.parse`` with a one-entry per-thread cache (symbols and imports share a parse)."""
    cached = getattr(_PARSE_CACHE, "entry", None)
    if cached is not None and cached[0] is text:
        tree: ast.Module | None = cached[1]
        return tree
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        tree = None
    _PARSE_CACHE.entry = (text, tree)
    return tree


def _py_func_sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    try:
        args = ast.unparse(node.args)
        ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    except Exception:  # unparse is best effort on unusual trees
        args, ret = "...", ""
    return _clean_sig(f"{prefix} {node.name}({args}){ret}")


def _py_class_sig(node: ast.ClassDef) -> str:
    try:
        bases = [ast.unparse(b) for b in node.bases] + [ast.unparse(k) for k in node.keywords]
    except Exception:
        bases = []
    return _clean_sig(f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}")


def _py_symbols_ast(tree: ast.Module, text: str) -> list[Symbol]:
    out: list[Symbol] = []
    src_lines = text.splitlines()

    def visit(body: list[ast.stmt], parent: str | None, in_class: bool) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                out.append(
                    Symbol(
                        name=node.name,
                        kind="method" if in_class else "function",
                        line=node.lineno,
                        end_line=node.end_lineno,
                        parent=parent,
                        signature=_py_func_sig(node),
                    )
                )
            elif isinstance(node, ast.ClassDef):
                out.append(
                    Symbol(
                        name=node.name,
                        kind="class",
                        line=node.lineno,
                        end_line=node.end_lineno,
                        parent=parent,
                        signature=_py_class_sig(node),
                    )
                )
                visit(node.body, node.name, True)
            elif isinstance(node, ast.Assign | ast.AnnAssign) and not in_class and parent is None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and _PY_CONST.match(target.id):
                        sig = src_lines[node.lineno - 1] if node.lineno <= len(src_lines) else target.id
                        out.append(
                            Symbol(
                                name=target.id,
                                kind="const",
                                line=node.lineno,
                                end_line=node.end_lineno,
                                signature=_clean_sig(sig),
                            )
                        )
            elif isinstance(node, ast.If | ast.With | ast.AsyncWith):
                visit(node.body, parent, in_class)
                visit(getattr(node, "orelse", []), parent, in_class)
            elif isinstance(node, ast.Try | ast.TryStar):
                visit(node.body, parent, in_class)
                for handler in node.handlers:
                    visit(handler.body, parent, in_class)
                visit(node.orelse, parent, in_class)
                visit(node.finalbody, parent, in_class)

    visit(tree.body, None, False)
    return out


_PY_DEF_RE = re.compile(r"^([ \t]*)(async[ \t]+)?def[ \t]+(\w+)[ \t]*\(", re.M)
_PY_CLASS_RE = re.compile(r"^([ \t]*)class[ \t]+(\w+)", re.M)
_PY_CONST_RE = re.compile(r"^([A-Z][A-Z0-9_]*)[ \t]*(?::[^=\n]*)?=(?!=)", re.M)


def _py_symbols_regex(text: str) -> list[Symbol]:
    lines = _LineIndex(text)
    found: list[tuple[int, int, str, str, str]] = []  # (pos, indent, kind, name, sig)
    for m in _PY_CLASS_RE.finditer(text):
        found.append((m.start(), len(m.group(1).expandtabs(4)), "class", m.group(2), lines.line_text_at(m.start())))
    for m in _PY_DEF_RE.finditer(text):
        found.append((m.start(), len(m.group(1).expandtabs(4)), "function", m.group(3), lines.line_text_at(m.start())))
    for m in _PY_CONST_RE.finditer(text):
        found.append((m.start(), 0, "const", m.group(1), lines.line_text_at(m.start())))
    found.sort()
    out: list[Symbol] = []
    class_stack: list[tuple[int, str]] = []  # (indent, name)
    for pos, indent, kind, name, sig in found:
        while class_stack and class_stack[-1][0] >= indent:
            class_stack.pop()
        parent = class_stack[-1][1] if class_stack else None
        if kind == "function" and parent is not None:
            kind = "method"
        if kind == "const" and parent is not None:
            continue
        out.append(Symbol(name=name, kind=kind, line=lines.line_of(pos), parent=parent, signature=_clean_sig(sig)))
        if kind == "class":
            class_stack.append((indent, name))
    return out


# --------------------------------------------------------------------------- brace scanning


class _LineIndex:
    def __init__(self, text: str) -> None:
        self.text = text
        self.starts = [0]
        self.starts.extend(m.end() for m in re.finditer("\n", text))

    def line_of(self, pos: int) -> int:
        return bisect.bisect_right(self.starts, pos)

    def line_text_at(self, pos: int) -> str:
        line = self.line_of(pos)
        start = self.starts[line - 1]
        end = self.starts[line] - 1 if line < len(self.starts) else len(self.text)
        return self.text[start:end]


_STR_DQ = r'"(?:\\.|[^"\\\n])*"'
_TRIPLE_DQ = r'"""[\s\S]*?(?:"""|\Z)'
_CHAR_LIT = r"'(?:\\.[^'\n]{0,10}|[^'\\\n])'"
_STR_SQ = r"'(?:\\.|[^'\\\n])*'"
_LINE_COMMENT = r"//[^\n]*"
_BLOCK_COMMENT = r"/\*[\s\S]*?(?:\*/|\Z)"
_STRUCT_TOKENS = r"[{}()\n;]"

_TOKENIZERS: dict[str, re.Pattern[str]] = {
    "js": re.compile(
        "|".join(
            [_LINE_COMMENT, _BLOCK_COMMENT, _STR_DQ, _STR_SQ, r"`(?:\\[\s\S]|[^`\\])*`", _STRUCT_TOKENS]
        )
    ),
    "go": re.compile("|".join([_LINE_COMMENT, _BLOCK_COMMENT, _STR_DQ, _CHAR_LIT, r"`[^`]*`", _STRUCT_TOKENS])),
    "c_like": re.compile(
        "|".join([_LINE_COMMENT, _BLOCK_COMMENT, _TRIPLE_DQ, _STR_DQ, _CHAR_LIT, _STRUCT_TOKENS])
    ),
}


@dataclass
class _Structure:
    """Brace/paren nesting information for a source text."""

    line_depth: list[int] = field(default_factory=list)  # 1-based: depth at line start
    line_pdepth: list[int] = field(default_factory=list)  # 1-based: paren depth at line start
    opens: list[int] = field(default_factory=list)  # positions of "{" in code
    open_depth: dict[int, int] = field(default_factory=dict)
    open_pdepth: dict[int, int] = field(default_factory=dict)
    close_of: dict[int, int] = field(default_factory=dict)
    semis: list[int] = field(default_factory=list)


def _scan(text: str, flavor: str) -> _Structure:
    st = _Structure(line_depth=[0, 0], line_pdepth=[0, 0])
    depth = 0
    pdepth = 0
    stack: list[int] = []
    for m in _TOKENIZERS[flavor].finditer(text):
        tok = m.group()
        if tok == "\n":
            st.line_depth.append(depth)
            st.line_pdepth.append(pdepth)
        elif tok == "{":
            pos = m.start()
            st.opens.append(pos)
            st.open_depth[pos] = depth
            st.open_pdepth[pos] = pdepth
            stack.append(pos)
            depth += 1
        elif tok == "}":
            if stack:
                st.close_of[stack.pop()] = m.start()
            depth = max(0, depth - 1)
        elif tok == "(":
            pdepth += 1
        elif tok == ")":
            pdepth = max(0, pdepth - 1)
        elif tok == ";":
            st.semis.append(m.start())
        else:
            for _ in range(tok.count("\n")):
                st.line_depth.append(depth)
                st.line_pdepth.append(pdepth)
    return st


def _match_paren(text: str, open_pos: int, limit: int = 4000) -> int | None:
    depth = 0
    end = min(len(text), open_pos + limit)
    for i in range(open_pos, end):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _body_after(
    text: str,
    st: _Structure,
    start: int,
    paren_at: int | None = None,
    limit: int = 3000,
    expr_check: bool = True,
    same_line: bool = False,
) -> tuple[int, int] | None:
    """Locate the ``{...}`` body that follows a declaration, if any.

    ``paren_at`` points at a parameter list to skip first. ``expr_check`` rejects
    expression-bodied declarations (``= expr``); ``same_line`` (languages without
    semicolons) requires the body to open on the line where the header ends.
    """
    search_from = start
    if paren_at is not None:
        close = _match_paren(text, paren_at)
        if close is None:
            return None
        search_from = close + 1
    i = bisect.bisect_left(st.opens, search_from)
    if i >= len(st.opens):
        return None
    open_pos = st.opens[i]
    if open_pos - start > limit:
        return None
    j = bisect.bisect_left(st.semis, search_from)
    if j < len(st.semis) and st.semis[j] < open_pos:
        return None
    between = text[search_from:open_pos]
    if expr_check and paren_at is not None and re.search(r"=(?!>)", between):
        return None  # expression-bodied declaration
    if "\n\n" in between.replace("\r", ""):
        return None
    if same_line and "\n" in between and re.search(r"[A-Za-z]", between[between.index("\n") :]):
        return None
    return open_pos, st.close_of.get(open_pos, len(text) - 1)


def _header_paren(text: str, pos: int) -> int | None:
    """Index of a parameter list right after a type name (skipping generics), if any."""
    i = pos
    n = len(text)
    while i < n and text[i] in " \t":
        i += 1
    if i < n and text[i] == "<":
        depth = 0
        while i < n:
            if text[i] == "<":
                depth += 1
            elif text[i] == ">":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            elif text[i] in "{;\n":
                return None
            i += 1
        while i < n and text[i] in " \t":
            i += 1
    return i if i < n and text[i] == "(" else None


@dataclass
class _Container:
    name: str
    kind: str | None  # None: not emitted as a symbol (Rust impl/mod blocks)
    open_pos: int
    close_pos: int
    body_depth: int
    body_pdepth: int
    is_module: bool = False  # functions inside are functions (Rust ``mod x { ... }``)


class _ContainerIndex:
    """Find the container whose body directly holds a position, in O(log n).

    Containers with the same body depth are siblings and never overlap, so the
    candidate is the last one at that depth that opened before the position.
    """

    def __init__(self, containers: list[_Container]) -> None:
        groups: dict[int, list[_Container]] = {}
        for c in containers:
            groups.setdefault(c.body_depth, []).append(c)
        self._by_depth: dict[int, tuple[list[int], list[_Container]]] = {}
        for depth, items in groups.items():
            items.sort(key=lambda c: c.open_pos)
            self._by_depth[depth] = ([c.open_pos for c in items], items)

    def find(self, pos: int, depth: int, pdepth: int) -> _Container | None:
        entry = self._by_depth.get(depth)
        if entry is None:
            return None
        opens, items = entry
        i = bisect.bisect_left(opens, pos) - 1
        if i < 0:
            return None
        candidate = items[i]
        if candidate.close_pos > pos and candidate.body_pdepth == pdepth:
            return candidate
        return None


@dataclass
class _LangSpec:
    flavor: str
    containers: list[tuple[re.Pattern[str], int, str | None]]  # regex, name group, kind (None: see below)
    functions: list[tuple[re.Pattern[str], int]]  # regex ending at "(", name group
    members: list[tuple[re.Pattern[str], int]] = field(default_factory=list)  # only inside containers
    others: list[tuple[re.Pattern[str], int, str]] = field(default_factory=list)  # top-level simple decls
    top_level_functions: bool = True
    keywords: frozenset[str] = frozenset()
    container_kind: dict[str, str] = field(default_factory=dict)  # keyword (group 1) -> kind
    module_containers: list[tuple[re.Pattern[str], int]] = field(default_factory=list)
    same_line_bodies: bool = False  # no statement terminators (Kotlin)


_JS_ID = r"[A-Za-z_$][\w$]*"
_JS_KEYWORDS = frozenset(
    {
        "if", "for", "while", "switch", "catch", "return", "function", "with", "else", "do", "try",
        "new", "typeof", "await", "yield", "super", "throw", "delete", "void", "in", "of", "case",
    }
)

_JS = _LangSpec(
    flavor="js",
    containers=[
        (
            re.compile(
                rf"^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?(?:declare[ \t]+)?(?:abstract[ \t]+)?class[ \t]+({_JS_ID})",
                re.M,
            ),
            1,
            "class",
        ),
        (re.compile(rf"^[ \t]*(?:export[ \t]+)?(?:declare[ \t]+)?interface[ \t]+({_JS_ID})", re.M), 1, "interface"),
        (
            re.compile(rf"^[ \t]*(?:export[ \t]+)?(?:declare[ \t]+)?(?:const[ \t]+)?enum[ \t]+({_JS_ID})", re.M),
            1,
            "enum",
        ),
    ],
    functions=[
        (
            re.compile(
                rf"^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?(?:declare[ \t]+)?(?:async[ \t]+)?function\b[ \t]*\*?[ \t]*"
                rf"({_JS_ID})[ \t]*(?:<[^>\n]*>)?[ \t]*\(",
                re.M,
            ),
            1,
        ),
    ],
    members=[
        (
            re.compile(
                r"^[ \t]*(?:@[\w.]+(?:\([^)\n]*\))?[ \t]+)*"
                r"(?:(?:public|private|protected|static|async|readonly|abstract|override|declare|get|set)[ \t]+)*"
                rf"\*?[ \t]*(#?{_JS_ID})[ \t]*\??[ \t]*(?:<[^>\n]*>)?[ \t]*\(",
                re.M,
            ),
            1,
        ),
        (
            re.compile(
                r"^[ \t]*(?:(?:public|private|protected|static|readonly|override)[ \t]+)*"
                rf"(#?{_JS_ID})[ \t]*(?::[^=\n]+)?=[ \t]*(?:async[ \t]+)?(?:\([^)\n]*\)|{_JS_ID})[ \t]*(?::[^=\n]+)?=>",
                re.M,
            ),
            1,
        ),
    ],
    keywords=_JS_KEYWORDS,
)

_JS_ARROW = re.compile(
    rf"^[ \t]*(?:export[ \t]+)?(?:const|let|var)[ \t]+({_JS_ID})[ \t]*(?::[^=\n]+)?=[ \t]*(?:async[ \t]+)?"
    rf"(?:function\b|(?:<[^>\n]*>)?\([^)]*\)[ \t]*(?::[^=\n]+)?=>|{_JS_ID}[ \t]*=>)",
    re.M,
)
_JS_CJS_EXPORT = re.compile(
    rf"^[ \t]*(?:module\.)?exports\.({_JS_ID})[ \t]*=[ \t]*(?:async[ \t]+)?"
    rf"(?:function\b|\([^)\n]*\)[ \t]*=>|{_JS_ID}[ \t]*=>)",
    re.M,
)
_JS_TYPE = re.compile(rf"^[ \t]*(?:export[ \t]+)?(?:declare[ \t]+)?type[ \t]+({_JS_ID})[ \t]*(?:<[^=\n]*>)?[ \t]*=", re.M)
_JS_CONST = re.compile(rf"^[ \t]*(export[ \t]+)?(const|let|var)[ \t]+({_JS_ID})\b", re.M)

_GO_FUNC = re.compile(
    r"^func[ \t]*(?:\([ \t]*(?:\w+[ \t]+)?\*?[ \t]*(\w+)(?:\[[^\]\n]*\])?[ \t]*\)[ \t]*)?(\w+)[ \t]*(?:\[[^\]\n]*\])?[ \t]*\(",
    re.M,
)
_GO_TYPE = re.compile(r"^type[ \t]+(\w+)(?:\[[^\]\n]*\])?[ \t]+(=[ \t]*)?(struct\b|interface\b)?", re.M)
_GO_GROUP = re.compile(r"^(type|const|var)[ \t]*\(", re.M)
_GO_GROUP_ITEM = re.compile(r"^[ \t]+(\w+)(?:\[[^\]\n]*\])?(?:[ \t]+(struct\b|interface\b))?", re.M)
_GO_SIMPLE = re.compile(r"^(const|var)[ \t]+(\w+)", re.M)

_RS_VIS = r"(?:pub(?:[ \t]*\([^)\n]*\))?[ \t]+)?"
_RUST = _LangSpec(
    flavor="c_like",
    containers=[
        (re.compile(rf"^[ \t]*{_RS_VIS}(?:unsafe[ \t]+)?(?:auto[ \t]+)?trait[ \t]+(\w+)", re.M), 1, "interface"),
        (
            re.compile(
                r"^[ \t]*(?:unsafe[ \t]+)?impl\b(?:[ \t]*<[^{\n]*?>)?[ \t]+"
                r"(?:!?[\w:]+(?:<[^{\n]*?>)?[ \t]+for[ \t]+)?&?(?:mut[ \t]+)?((?:\w+::)*\w+)",
                re.M,
            ),
            1,
            None,
        ),
    ],
    module_containers=[(re.compile(rf"^[ \t]*{_RS_VIS}mod[ \t]+(\w+)[ \t]*\{{", re.M), 1)],
    functions=[
        (
            re.compile(
                rf"^[ \t]*{_RS_VIS}(?:default[ \t]+)?(?:const[ \t]+)?(?:async[ \t]+)?(?:unsafe[ \t]+)?"
                r"(?:extern[ \t]+(?:\"[^\"\n]*\"[ \t]+)?)?fn[ \t]+(\w+)[ \t]*(?:<[^(\n]*>)?[ \t]*\(",
                re.M,
            ),
            1,
        ),
    ],
    others=[
        (re.compile(rf"^[ \t]*{_RS_VIS}struct[ \t]+(\w+)", re.M), 1, "struct"),
        (re.compile(rf"^[ \t]*{_RS_VIS}union[ \t]+(\w+)", re.M), 1, "struct"),
        (re.compile(rf"^[ \t]*{_RS_VIS}enum[ \t]+(\w+)", re.M), 1, "enum"),
        (re.compile(rf"^[ \t]*{_RS_VIS}type[ \t]+(\w+)", re.M), 1, "type"),
        (re.compile(rf"^[ \t]*{_RS_VIS}(?:const|static)[ \t]+(?:mut[ \t]+)?([A-Z_][A-Z0-9_]*)[ \t]*:", re.M), 1, "const"),
    ],
)

_JAVA_KEYWORDS = frozenset(
    {
        "return", "new", "throw", "else", "case", "await", "yield", "goto", "using", "is", "as", "in",
        "out", "ref", "if", "for", "while", "switch", "catch", "do", "try", "synchronized", "assert",
        "var", "let", "val", "import", "package", "public", "private", "protected", "static",
        "record", "class", "interface", "enum", "struct", "throws", "extends", "implements",
    }
)
_JAVA_MODS = (
    r"(?:(?:public|private|protected|internal|static|final|abstract|synchronized|native|default|override|"
    r"virtual|async|sealed|extern|unsafe|new|partial|readonly|strictfp|transient|volatile)[ \t]+)*"
)
_ANNOT = r"(?:(?:@[\w.]+(?:\([^)\n]*\))?|\[[^\]\n]*\])[ \t]+)*"
_JAVA_METHOD = re.compile(
    rf"^[ \t]*{_ANNOT}{_JAVA_MODS}(?:<[^>\n]+>[ \t]+)?"
    r"([\w.$]+(?:<[^()\n]*?>)?(?:\[\])*\??)[ \t]+(\w+)[ \t]*(?:<[^>\n]*>)?[ \t]*\(",
    re.M,
)
_JAVA_CTOR = re.compile(rf"^[ \t]*{_ANNOT}(?:(?:public|private|protected|internal)[ \t]+)(\w+)[ \t]*\(", re.M)

_JAVA = _LangSpec(
    flavor="c_like",
    containers=[
        (
            re.compile(
                rf"^[ \t]*{_ANNOT}(?:(?:public|private|protected|static|final|abstract|sealed|non-sealed|strictfp)[ \t]+)*"
                r"(class|interface|enum|record|@interface)[ \t]+(\w+)",
                re.M,
            ),
            2,
            None,
        ),
    ],
    functions=[],
    top_level_functions=False,
    keywords=_JAVA_KEYWORDS,
    container_kind={"class": "class", "interface": "interface", "enum": "enum", "record": "class", "@interface": "interface"},
)

_CSHARP = _LangSpec(
    flavor="c_like",
    containers=[
        (
            re.compile(
                rf"^[ \t]*{_ANNOT}(?:(?:public|private|protected|internal|static|sealed|abstract|partial|readonly|"
                r"unsafe|new|file|ref)[ \t]+)*(class|interface|enum|struct|record)(?:[ \t]+(?:class|struct))?[ \t]+(\w+)",
                re.M,
            ),
            2,
            None,
        ),
    ],
    functions=[],
    top_level_functions=False,
    keywords=_JAVA_KEYWORDS,
    container_kind={"class": "class", "interface": "interface", "enum": "enum", "struct": "struct", "record": "class"},
)

_KOTLIN = _LangSpec(
    flavor="c_like",
    containers=[
        (
            re.compile(
                rf"^[ \t]*{_ANNOT}(?:(?:public|private|protected|internal|abstract|open|final|sealed|data|enum|annotation|"
                r"inner|value|inline|companion|expect|actual|fun)[ \t]+)*(class|interface|object)[ \t]+(\w+)",
                re.M,
            ),
            2,
            None,
        ),
    ],
    functions=[
        (
            re.compile(
                rf"^[ \t]*{_ANNOT}(?:(?:public|private|protected|internal|open|override|abstract|final|suspend|inline|"
                r"operator|infix|tailrec|external|actual|expect)[ \t]+)*fun[ \t]+(?:<[^>\n]+>[ \t]*)?"
                r"(?:[\w.<>?, ]+\.)?(\w+)[ \t]*\(",
                re.M,
            ),
            1,
        ),
    ],
    others=[(re.compile(r"^(?:(?:public|private|internal)[ \t]+)?const[ \t]+val[ \t]+(\w+)", re.M), 1, "const")],
    container_kind={"class": "class", "interface": "interface", "object": "class"},
    same_line_bodies=True,
)


def _brace_symbols(text: str, spec: _LangSpec, language: str) -> list[Symbol]:
    st = _scan(text, spec.flavor)
    lines = _LineIndex(text)
    out: list[tuple[int, Symbol]] = []
    containers: list[_Container] = []

    def depth_at(pos: int) -> tuple[int, int]:
        line = lines.line_of(pos)
        if line < len(st.line_depth):
            return st.line_depth[line], st.line_pdepth[line]
        return 0, 0

    def end_line(body: tuple[int, int] | None) -> int | None:
        return lines.line_of(body[1]) if body else None

    for regex, group in spec.module_containers:
        for m in regex.finditer(text):
            body = _body_after(text, st, m.end() - 1)
            if body is not None:
                containers.append(
                    _Container(
                        name=m.group(group),
                        kind=None,
                        open_pos=body[0],
                        close_pos=body[1],
                        body_depth=st.open_depth.get(body[0], 0) + 1,
                        body_pdepth=st.open_pdepth.get(body[0], 0),
                        is_module=True,
                    )
                )

    declared: list[tuple[re.Match[str], str, str, tuple[int, int] | None]] = []
    for regex, group, kind in spec.containers:
        for m in regex.finditer(text):
            name = m.group(group)
            body = _body_after(
                text,
                st,
                m.end(),
                paren_at=_header_paren(text, m.end()),
                expr_check=False,
                same_line=spec.same_line_bodies,
            )
            emitted_kind = kind
            if kind is None and spec.container_kind:
                word = m.group(1)
                emitted_kind = spec.container_kind.get(word, "class")
                if language == "kotlin" and "enum" in m.group(0).split(word)[0].split():
                    emitted_kind = "enum"
            if body is not None:
                containers.append(
                    _Container(
                        name=name.split("::")[-1],
                        kind=emitted_kind,
                        open_pos=body[0],
                        close_pos=body[1],
                        body_depth=st.open_depth.get(body[0], 0) + 1,
                        body_pdepth=st.open_pdepth.get(body[0], 0),
                    )
                )
            if emitted_kind is not None:
                declared.append((m, name, emitted_kind, body))

    index = _ContainerIndex(containers)
    for m, name, emitted_kind, body in declared:
        d, pd = depth_at(m.start())
        parent = index.find(m.start(), d, pd)
        out.append(
            (
                m.start(),
                Symbol(
                    name=name,
                    kind=emitted_kind,
                    line=lines.line_of(m.start()),
                    end_line=end_line(body),
                    parent=parent.name if parent else None,
                    signature=_clean_sig(lines.line_text_at(m.start())),
                ),
            )
        )

    seen_pos: set[int] = set()

    def add_function(m: re.Match[str], group: int, members_only: bool) -> None:
        name = m.group(group).lstrip("#")
        if name in spec.keywords or m.start() in seen_pos:
            return
        d, pd = depth_at(m.start())
        parent = index.find(m.start(), d, pd)
        if parent is None:
            if members_only or not spec.top_level_functions or d != 0 or pd != 0:
                return
        paren_at = m.end() - 1 if m.group(0).endswith("(") else None
        body = _body_after(text, st, m.start(), paren_at=paren_at, same_line=spec.same_line_bodies)
        seen_pos.add(m.start())
        kind = "method" if parent is not None and not parent.is_module else "function"
        out.append(
            (
                m.start(),
                Symbol(
                    name=name,
                    kind=kind,
                    line=lines.line_of(m.start()),
                    end_line=end_line(body),
                    parent=parent.name if parent else None,
                    signature=_clean_sig(lines.line_text_at(m.start())),
                ),
            )
        )

    for regex, group in spec.functions:
        for m in regex.finditer(text):
            add_function(m, group, members_only=False)
    for regex, group in spec.members:
        for m in regex.finditer(text):
            add_function(m, group, members_only=True)
    if language in ("java", "csharp"):
        names = {c.name for c in containers}
        for m in _JAVA_CTOR.finditer(text):
            if m.group(1) in names:
                add_function(m, 1, members_only=True)
        for m in _JAVA_METHOD.finditer(text):
            if m.group(1) not in spec.keywords:
                add_function(m, 2, members_only=True)

    for regex, group, kind in spec.others:
        for m in regex.finditer(text):
            d, pd = depth_at(m.start())
            parent = index.find(m.start(), d, pd)
            if parent is None and d != 0:
                continue
            body = _body_after(text, st, m.end()) if kind in ("struct", "enum") else None
            out.append(
                (
                    m.start(),
                    Symbol(
                        name=m.group(group),
                        kind=kind,
                        line=lines.line_of(m.start()),
                        end_line=end_line(body),
                        parent=parent.name if parent else None,
                        signature=_clean_sig(lines.line_text_at(m.start())),
                    ),
                )
            )

    if language in ("javascript", "typescript"):
        out.extend(_js_extra(text, st, lines, seen_pos, depth_at))

    out.sort(key=lambda item: (item[0], item[1].name))
    return [sym for _, sym in out]


def _js_extra(
    text: str,
    st: _Structure,
    lines: _LineIndex,
    seen_pos: set[int],
    get_depth: Callable[[int], tuple[int, int]],
) -> list[tuple[int, Symbol]]:
    out: list[tuple[int, Symbol]] = []
    taken: set[int] = set()

    for m in _JS_ARROW.finditer(text):
        if get_depth(m.start()) != (0, 0):
            continue
        taken.add(m.start())
        arrow = text.find("=>", m.start(), m.end() + 1)
        body: tuple[int, int] | None = None
        if arrow != -1:
            k = arrow + 2
            while k < len(text) and text[k] in " \t\r\n":
                k += 1
            if k < len(text) and text[k] == "{" and k in st.close_of:
                body = (k, st.close_of[k])
        else:
            paren = text.find("(", m.start(), m.end() + 200)
            body = _body_after(text, st, m.start(), paren_at=paren if paren != -1 else None)
        out.append(
            (
                m.start(),
                Symbol(
                    name=m.group(1),
                    kind="function",
                    line=lines.line_of(m.start()),
                    end_line=lines.line_of(body[1]) if body else None,
                    signature=_clean_sig(lines.line_text_at(m.start())),
                ),
            )
        )
    for m in _JS_CJS_EXPORT.finditer(text):
        if get_depth(m.start()) != (0, 0):
            continue
        out.append(
            (
                m.start(),
                Symbol(
                    name=m.group(1),
                    kind="function",
                    line=lines.line_of(m.start()),
                    signature=_clean_sig(lines.line_text_at(m.start())),
                ),
            )
        )
    for m in _JS_TYPE.finditer(text):
        if get_depth(m.start())[0] != 0:
            continue
        out.append(
            (
                m.start(),
                Symbol(
                    name=m.group(1),
                    kind="type",
                    line=lines.line_of(m.start()),
                    signature=_clean_sig(lines.line_text_at(m.start())),
                ),
            )
        )
    for m in _JS_CONST.finditer(text):
        if m.start() in taken or m.start() in seen_pos or get_depth(m.start()) != (0, 0):
            continue
        exported, keyword, name = m.group(1), m.group(2), m.group(3)
        is_upper = bool(_PY_CONST.match(name))
        if not exported and not is_upper:
            continue
        out.append(
            (
                m.start(),
                Symbol(
                    name=name,
                    kind="const" if keyword == "const" else "variable",
                    line=lines.line_of(m.start()),
                    signature=_clean_sig(lines.line_text_at(m.start())),
                ),
            )
        )
    return out


def _go_symbols(text: str) -> list[Symbol]:
    st = _scan(text, "go")
    lines = _LineIndex(text)
    out: list[tuple[int, Symbol]] = []

    def depth0(pos: int) -> bool:
        line = lines.line_of(pos)
        return line >= len(st.line_depth) or st.line_depth[line] == 0

    for m in _GO_FUNC.finditer(text):
        receiver, name = m.group(1), m.group(2)
        body = _body_after(text, st, m.start(), paren_at=m.end() - 1)
        out.append(
            (
                m.start(),
                Symbol(
                    name=name,
                    kind="method" if receiver else "function",
                    line=lines.line_of(m.start()),
                    end_line=lines.line_of(body[1]) if body else None,
                    parent=receiver,
                    signature=_clean_sig(lines.line_text_at(m.start())),
                ),
            )
        )
    for m in _GO_TYPE.finditer(text):
        name, alias, what = m.group(1), m.group(2), m.group(3)
        kind = "type" if alias or not what else ("struct" if what.startswith("struct") else "interface")
        body = _body_after(text, st, m.end()) if what and not alias else None
        out.append(
            (
                m.start(),
                Symbol(
                    name=name,
                    kind=kind,
                    line=lines.line_of(m.start()),
                    end_line=lines.line_of(body[1]) if body else None,
                    signature=_clean_sig(lines.line_text_at(m.start())),
                ),
            )
        )
    for m in _GO_SIMPLE.finditer(text):
        out.append(
            (
                m.start(),
                Symbol(
                    name=m.group(2),
                    kind="const" if m.group(1) == "const" else "variable",
                    line=lines.line_of(m.start()),
                    signature=_clean_sig(lines.line_text_at(m.start())),
                ),
            )
        )
    for g in _GO_GROUP.finditer(text):
        close = _match_paren(text, g.end() - 1, limit=200_000)
        if close is None:
            continue
        group_kind = g.group(1)
        for item in _GO_GROUP_ITEM.finditer(text, g.end(), close):
            if not depth0(item.start()) or _paren_depth(text, g.end(), item.start()) != 0:
                continue
            name, what = item.group(1), item.group(2)
            if group_kind == "type":
                kind = "struct" if what == "struct" else "interface" if what == "interface" else "type"
            else:
                kind = "const" if group_kind == "const" else "variable"
            out.append(
                (
                    item.start(),
                    Symbol(
                        name=name,
                        kind=kind,
                        line=lines.line_of(item.start()),
                        signature=_clean_sig(lines.line_text_at(item.start())),
                    ),
                )
            )
    out.sort(key=lambda item: (item[0], item[1].name))
    return [sym for _, sym in out]


def _paren_depth(text: str, start: int, end: int) -> int:
    segment = text[start:end]
    return segment.count("(") - segment.count(")")


def _rust_symbols(text: str) -> list[Symbol]:
    return _brace_symbols(text, _RUST, "rust")


_SPECS: dict[str, _LangSpec] = {
    "javascript": _JS,
    "typescript": _JS,
    "java": _JAVA,
    "csharp": _CSHARP,
    "kotlin": _KOTLIN,
}


def extract_symbols(path: str, text: str, language: str | None) -> list[Symbol]:
    """Extract definitions from ``text``. Unknown languages yield ``[]``."""
    if not text:
        return []
    try:
        if language == "python":
            tree = _parse_python(text)
            if tree is None:
                return _py_symbols_regex(text)
            return _py_symbols_ast(tree, text)
        if language == "go":
            return _go_symbols(text)
        if language == "rust":
            return _rust_symbols(text)
        spec = _SPECS.get(language or "")
        if spec is not None:
            return _brace_symbols(text, spec, language or "")
    except RecursionError:
        return []
    return []


# --------------------------------------------------------------------------- imports


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        if item and item not in seen:
            seen[item] = None
    return list(seen)


def _py_package_parts(path: str) -> list[str]:
    parts = path.replace("\\", "/").split("/")[:-1]
    if parts and parts[0] == "src":
        parts = parts[1:]
    return parts


def _py_imports_ast(path: str, tree: ast.Module) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.extend(_py_from_import(path, node.level, node.module, [a.name for a in node.names]))
    return out


def _py_from_import(path: str, level: int, module: str | None, names: list[str]) -> list[str]:
    if level:
        package = _py_package_parts(path)
        up = level - 1
        if up > len(package):
            return []
        base_parts = package[: len(package) - up] if up else package
        parts = base_parts + (module.split(".") if module else [])
    else:
        parts = module.split(".") if module else []
    if not parts:  # ``from . import x`` in a top-level module: siblings
        return [name for name in names if name and name != "*"] if level else []
    base = ".".join(parts)
    out = [base]
    out.extend(f"{base}.{name}" for name in names if name and name != "*")
    return out


_PY_IMPORT_RE = re.compile(r"^[ \t]*import[ \t]+([\w.]+(?:[ \t]+as[ \t]+\w+)?(?:[ \t]*,[ \t]*[\w.]+(?:[ \t]+as[ \t]+\w+)?)*)", re.M)
_PY_FROM_RE = re.compile(r"^[ \t]*from[ \t]+(\.*)([\w.]*)[ \t]+import[ \t]+\(?([\w ,\t*]+)", re.M)


def _py_imports_regex(path: str, text: str) -> list[str]:
    found: list[tuple[int, list[str]]] = []
    for m in _PY_IMPORT_RE.finditer(text):
        mods = [part.split()[0] for part in m.group(1).split(",") if part.strip()]
        found.append((m.start(), mods))
    for m in _PY_FROM_RE.finditer(text):
        names = [n.strip().split()[0] for n in m.group(3).split(",") if n.strip()]
        found.append((m.start(), _py_from_import(path, len(m.group(1)), m.group(2) or None, names)))
    found.sort(key=lambda item: item[0])
    return [mod for _, mods in found for mod in mods]


_JS_IMPORT_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"""\bimport\s+(?:type\s+)?(?:[^'";]*?\bfrom\s*)?['"]([^'"\n]+)['"]"""),
    re.compile(r"""\bexport\s+(?:type\s+)?(?:\*(?:\s+as\s+[\w$]+)?|\{[^}]*\})\s*from\s*['"]([^'"\n]+)['"]"""),
    re.compile(r"""\bimport\s*\(\s*['"]([^'"\n]+)['"]\s*\)"""),
    re.compile(r"""\brequire\s*\(\s*['"]([^'"\n]+)['"]\s*\)"""),
)

_GO_IMPORT_SINGLE = re.compile(r'^import[ \t]+(?:[\w.]+[ \t]+)?"([^"\n]+)"', re.M)
_GO_IMPORT_BLOCK = re.compile(r"^import[ \t]*\((.*?)^\)", re.M | re.S)
_GO_IMPORT_ITEM = re.compile(r'"([^"\n]+)"')

_RS_USE = re.compile(r"^[ \t]*(?:pub(?:[ \t]*\([^)\n]*\))?[ \t]+)?use[ \t]+([^;]+);", re.M)
_RS_MOD = re.compile(r"^[ \t]*(?:#\[[^\]\n]*\][ \t]*)*(?:pub(?:[ \t]*\([^)\n]*\))?[ \t]+)?mod[ \t]+(\w+)[ \t]*;", re.M)
_RS_AS = re.compile(r"\s+as\s+\w+")

_JAVA_IMPORT = re.compile(r"^[ \t]*import[ \t]+(?:static[ \t]+)?([\w.]+(?:\.\*)?)[ \t]*;", re.M)
_KOTLIN_IMPORT = re.compile(r"^[ \t]*import[ \t]+([\w.]+(?:\.\*)?)", re.M)
_CSHARP_USING = re.compile(r"^[ \t]*(?:global[ \t]+)?using[ \t]+(?:static[ \t]+)?(?:\w+[ \t]*=[ \t]*)?([\w.]+)[ \t]*;", re.M)
_C_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"\n]+)"', re.M)


def _rust_use_paths(raw: str) -> list[str]:
    cleaned = re.sub(r"\s+", "", _RS_AS.sub("", raw))
    if "{" not in cleaned:
        cleaned = cleaned.removesuffix("::*")
        return [cleaned] if cleaned else []
    prefix = cleaned[: cleaned.index("{")].rstrip(":")
    inner = cleaned[cleaned.index("{") + 1 : cleaned.rindex("}")] if "}" in cleaned else ""
    out = [prefix] if prefix else []
    depth = 0
    current = ""
    items: list[str] = []
    for ch in inner:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(current)
            current = ""
        else:
            current += ch
    items.append(current)
    for item in items:
        item = item.split("{")[0].rstrip(":").removesuffix("::*")
        if not item or item == "*":
            continue
        if item == "self":
            continue
        out.append(f"{prefix}::{item}" if prefix else item)
    return out


def extract_imports(path: str, text: str, language: str | None) -> list[str]:
    """Return imported module specifiers, in source order, de-duplicated.

    Python relative imports are resolved to absolute dotted names (``src/`` is
    treated as a source root), and ``from a import b`` also yields ``a.b``.
    Rust ``mod x;`` is reported as ``self::x``.
    """
    if not text:
        return []
    norm = path.replace("\\", "/")
    if language == "python":
        tree = _parse_python(text)
        if tree is None:
            return _dedupe(_py_imports_regex(norm, text))
        try:
            return _dedupe(_py_imports_ast(norm, tree))
        except RecursionError:
            return _dedupe(_py_imports_regex(norm, text))
    found: list[tuple[int, str]] = []
    if language in ("javascript", "typescript", "vue", "svelte"):
        for regex in _JS_IMPORT_RES:
            found.extend((m.start(), m.group(1)) for m in regex.finditer(text))
    elif language == "go":
        found.extend((m.start(), m.group(1)) for m in _GO_IMPORT_SINGLE.finditer(text))
        for block in _GO_IMPORT_BLOCK.finditer(text):
            found.extend((m.start(), m.group(1)) for m in _GO_IMPORT_ITEM.finditer(text, block.start(1), block.end(1)))
    elif language == "rust":
        for m in _RS_USE.finditer(text):
            found.extend((m.start(), p) for p in _rust_use_paths(m.group(1)))
        found.extend((m.start(), f"self::{m.group(1)}") for m in _RS_MOD.finditer(text))
    elif language in ("java", "scala", "groovy"):
        found.extend((m.start(), m.group(1)) for m in _JAVA_IMPORT.finditer(text))
        if language != "java":
            found.extend((m.start(), m.group(1)) for m in _KOTLIN_IMPORT.finditer(text))
    elif language == "kotlin":
        found.extend((m.start(), m.group(1)) for m in _KOTLIN_IMPORT.finditer(text))
    elif language == "csharp":
        found.extend((m.start(), m.group(1)) for m in _CSHARP_USING.finditer(text))
    elif language in ("c", "cpp", "objective-c"):
        found.extend((m.start(), m.group(1)) for m in _C_INCLUDE.finditer(text))
    found.sort(key=lambda item: item[0])
    return _dedupe([spec.strip() for _, spec in found])


def module_dir(path: str) -> str:
    """POSIX directory of a workspace-relative path ("" for the root)."""
    return posixpath.dirname(path.replace("\\", "/"))
