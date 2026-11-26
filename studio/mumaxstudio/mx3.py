"""Translator from the mumax3 script language (.mx3, Go-like syntax) to Python.

Handles: // and /* */ comments, ``:=``, ``++``/``--``, ``for init; cond; post {}``, ``for cond {}``, ``if / else if /
else``, ``func name(args) type {}``, ``&& || !``, ``true/false``, case-insensitive API identifiers, and mumax3's lazy
time-dependent expressions (``B_ext = vector(0, 0, 0.01*sin(2*pi*1e9*t))`` becomes ``lambda t: ...``).
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"""
    (?P<str>"(?:\\.|[^"\\])*"|`[^`]*`)
  | (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<id>[A-Za-z_À-￿][A-Za-z0-9_À-￿]*)
  | (?P<op>:=|\+\+|--|&&|\|\||==|!=|<=|>=|\+=|-=|\*=|/=|[-+*/%<>=!(),.\[\]{};:^&|])
  | (?P<ws>\s+)
""", re.X)


class MX3Error(SyntaxError):
    pass


def tokenize(expr):
    pos, out = 0, []
    while pos < len(expr):
        mm = _TOKEN.match(expr, pos)
        if not mm:
            raise MX3Error(f"unexpected character {expr[pos]!r} in: {expr}")
        kind = mm.lastgroup
        if kind != "ws":
            out.append((kind, mm.group()))
        pos = mm.end()
    return out


def strip_comments(src):
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif c == "`":
            j = src.find("`", i + 1)
            j = n if j < 0 else j
            out.append(src[i:j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            seg = src[i:(n if j < 0 else j + 2)]
            out.append("\n" * seg.count("\n"))
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def scan(src):
    """Split into items: ('stmt', text, line), ('open', line), ('close', line)."""
    src = strip_comments(src)
    items, buf, depth, line, start_line = [], [], 0, 1, 1
    i, n = 0, len(src)

    def flush():
        txt = "".join(buf).strip()
        if txt:
            items.append(("stmt", txt, start_line))
        buf.clear()

    while i < n:
        c = src[i]
        if not buf or not "".join(buf).strip():
            start_line = line
        if c in "\"`":
            q = c
            j = i + 1
            while j < n and src[j] != q:
                j += 2 if (q == '"' and src[j] == "\\") else 1
            buf.append(src[i:j + 1])
            line += src[i:j + 1].count("\n")
            i = j + 1
            continue
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if depth == 0 and c == "{":
            flush()
            items.append(("open", line))
        elif depth == 0 and c == "}":
            flush()
            items.append(("close", line))
        elif depth == 0 and c == ";" and not re.match(r"\s*for\b", "".join(buf)):
            flush()
        elif depth == 0 and c == "\n":
            # allow line continuation after an operator or comma
            txt = "".join(buf).rstrip()
            if txt and txt[-1] in "+-*/,(&|=<>" and not txt.endswith("++") and not txt.endswith("--"):
                buf.append(" ")
            else:
                flush()
        else:
            buf.append(c)
        if c == "\n":
            line += 1
        i += 1
    flush()
    return items


class Translator:
    def __init__(self, api_names, params=(), lazy_methods=("setregion", "add", "set")):
        self.api = {k.lower(): k for k in api_names}
        self.params = {p.lower() for p in params}
        self.scopes = [{}]  # block scopes of user variables declared with :=, lower-case name -> python name
        self.lazy_methods = lazy_methods
        self.lines = []  # (python line, source line)

    # ----------------------------------------------------------------- expressions
    def expr(self, text):
        toks = tokenize(text)
        return self._join(self._conv(toks))

    def _conv(self, toks):
        out = []
        for k, (kind, v) in enumerate(toks):
            prev = toks[k - 1][1] if k > 0 else None
            if kind == "op":
                if v == "&&":
                    v = " and "
                elif v == "||":
                    v = " or "
                elif v == "!":
                    v = " not "
                elif v == "^":
                    v = "**"  # mumax script has no xor; ^ is (rarely) used as power
                out.append(("op", v))
            elif kind == "id":
                low = v.lower()
                if prev == ".":
                    out.append(("id", v))
                elif v in ("true", "false", "nil"):
                    out.append(("id", {"true": "True", "false": "False", "nil": "None"}[v]))
                elif self.lookup(low) is not None:
                    out.append(("id", self.lookup(low)))
                elif low in self.api:
                    out.append(("id", self.api[low]))
                else:
                    out.append(("id", v))
            elif kind == "str":
                if v.startswith("`"):
                    v = repr(v[1:-1])
                out.append(("str", v))
            else:
                out.append((kind, v))
        return out

    def declare(self, name, outer=False):
        """Declare a user variable (x := ...). Names that collide with a mumax3 name (e.g. j vs the current J)
        shadow it inside their block, as in mumax3: they get a private Python name."""
        low = name.lower()
        py = "_u_" + name if low in self.api else name
        self.scopes[-2 if outer and len(self.scopes) > 1 else -1][low] = py
        return py

    def lookup(self, low):
        for sc in reversed(self.scopes):
            if low in sc:
                return sc[low]
        return None

    @staticmethod
    def _join(toks):
        s = []
        for kind, v in toks:
            if s and kind in ("id", "num", "str") and s[-1][0] in ("id", "num", "str"):
                s.append(("ws", " "))
            s.append((kind, v))
        return "".join(v for _, v in s)

    @staticmethod
    def _uses_t(toks):
        return any(kind == "id" and v in ("t", "T") for kind, v in toks)

    @staticmethod
    def _split_args(toks):
        """Split tokens of an argument list on top-level commas."""
        args, cur, depth = [], [], 0
        for tk in toks:
            if tk[1] in ("(", "["):
                depth += 1
            elif tk[1] in (")", "]"):
                depth -= 1
            if tk[1] == "," and depth == 0:
                args.append(cur)
                cur = []
            else:
                cur.append(tk)
        args.append(cur)
        return args

    def _lazy(self, toks, nargs=1):
        body = self._join(self._conv(toks))
        if nargs == 0:
            return f"(lambda: ({body}))"
        return f"(lambda t: ({body}))"

    # ----------------------------------------------------------------- statements
    def statement(self, text):
        toks = tokenize(text)
        if not toks:
            return []
        # x++ / x--
        if len(toks) >= 2 and toks[-1][1] in ("++", "--"):
            target = self._join(self._conv(toks[:-1]))
            return [f"{target} {'+' if toks[-1][1] == '++' else '-'}= 1"]
        # return / break / continue
        if toks[0] == ("id", "return"):
            return ["return " + self._join(self._conv(toks[1:]))] if len(toks) > 1 else ["return"]
        if toks[0][1] in ("break", "continue") and len(toks) == 1:
            return [toks[0][1]]
        # assignment
        depth = 0
        for k, (kind, v) in enumerate(toks):
            if v in ("(", "["):
                depth += 1
            elif v in (")", "]"):
                depth -= 1
            elif depth == 0 and kind == "op" and v in (":=", "=", "+=", "-=", "*=", "/="):
                lhs, rhs = toks[:k], toks[k + 1:]
                if v == ":=":
                    for kk, vv in lhs:
                        if kk == "id":
                            self.declare(vv)
                    v = "="
                lhs_s = self._join(self._conv(lhs))
                single = len(lhs) == 1 and lhs[0][0] == "id"
                if single and v == "=" and lhs[0][1].lower() in self.params and self._uses_t(rhs) \
                        and self.lookup(lhs[0][1].lower()) is None:
                    return [f"{lhs_s} = {self._lazy(rhs)}"]
                return [f"{lhs_s} {v} {self._join(self._conv(rhs))}"]
        # call statements with lazy arguments
        out = self._lazy_call(toks)
        if out is not None:
            return [out]
        return [self._join(self._conv(toks))]

    def _lazy_call(self, toks):
        # RunWhile(cond)  /  TableAddVar(expr, name, unit)
        if len(toks) >= 3 and toks[0][0] == "id" and toks[1][1] == "(" and toks[-1][1] == ")":
            fname = toks[0][1].lower()
            if fname in ("runwhile", "tableaddvar", "tableaddvariable"):
                args = self._split_args(toks[2:-1])
                first = self._lazy(args[0], nargs=0)
                rest = [self._join(self._conv(a)) for a in args[1:]]
                canon = self.api.get(fname, toks[0][1])
                return f"{canon}({', '.join([first] + rest)})"
        # Param.SetRegion(r, expr(t)) / Param.Add(mask, expr(t))
        if len(toks) >= 5 and toks[0][0] == "id" and toks[1][1] == "." and toks[2][0] == "id" and toks[3][1] == "(" \
                and toks[-1][1] == ")" and toks[0][1].lower() in self.params and toks[2][1].lower() in self.lazy_methods:
            args = self._split_args(toks[4:-1])
            conv = []
            for i, a in enumerate(args):
                if i == len(args) - 1 and self._uses_t(a):
                    conv.append(self._lazy(a))
                else:
                    conv.append(self._join(self._conv(a)))
            obj = self._join(self._conv(toks[:1]))
            return f"{obj}.{toks[2][1]}({', '.join(conv)})"
        return None

    def header(self, text):
        """Translate a block header; returns (python header lines, post statement lines or None)."""
        mm = re.match(r"for\b(.*)$", text, re.S)
        if mm:
            rest = mm.group(1).strip()
            if not rest:
                return ["while True:"], None, []
            parts = [p.strip() for p in rest.split(";")]
            if len(parts) == 3:
                init = self.statement(parts[0]) if parts[0] else []
                cond = self.expr(parts[1]) if parts[1] else "True"
                post = self.statement(parts[2]) if parts[2] else []
                return [f"while {cond}:"], post, init
            return [f"while {self.expr(rest)}:"], None, []
        mm = re.match(r"else\s+if\b(.*)$", text, re.S)
        if mm:
            return [f"elif {self.expr(mm.group(1))}:"], None, []
        if text.strip() == "else":
            return ["else:"], None, []
        mm = re.match(r"if\b(.*)$", text, re.S)
        if mm:
            return [f"if {self.expr(mm.group(1))}:"], None, []
        mm = re.match(r"func\s+([A-Za-z_]\w*)\s*\((.*?)\)(.*)$", text, re.S)
        if mm:
            name = self.declare(mm.group(1), outer=True)
            params = []
            for p in mm.group(2).split(","):
                p = p.strip()
                if p:
                    params.append(self.declare(p.split()[0]))
            return [f"def {name}({', '.join(params)}):"], None, []
        raise MX3Error(f"unsupported block statement: {text}")

    # ----------------------------------------------------------------- program
    def translate(self, src):
        items = scan(src)
        lines = []
        stack = []  # (post lines, has_body)
        level = 0
        i = 0
        pending_else = False
        while i < len(items):
            it = items[i]
            nxt = items[i + 1] if i + 1 < len(items) else None
            if it[0] == "stmt" and nxt is not None and nxt[0] == "open":
                text, ln = it[1], it[2]
                self.scopes.append({})
                hdr, post, init = self.header(text)
                for s in init:
                    lines.append(("    " * level + s, ln))
                for s in hdr:
                    lines.append(("    " * level + s, ln))
                stack.append([post, False, ln])
                level += 1
                i += 2
                pending_else = False
                continue
            if it[0] == "open":  # bare block
                self.scopes.append({})
                lines.append(("    " * level + "if True:", it[1]))
                stack.append([None, False, it[1]])
                level += 1
                i += 1
                continue
            if it[0] == "close":
                if not stack:
                    raise MX3Error(f"line {it[1]}: unmatched }}")
                post, has_body, ln = stack.pop()
                if len(self.scopes) > 1:
                    self.scopes.pop()
                if post:
                    for s in post:
                        lines.append(("    " * level + s, it[1]))
                    has_body = True
                if not has_body:
                    lines.append(("    " * level + "pass", it[1]))
                level -= 1
                if stack:
                    stack[-1][1] = True
                pending_else = True
                i += 1
                continue
            text, ln = it[1], it[2]
            try:
                out = self.statement(text)
            except MX3Error as e:
                raise MX3Error(f"line {ln}: {e}") from None
            for s in out:
                lines.append(("    " * level + s, ln))
            if stack:
                stack[-1][1] = True
            i += 1
        del pending_else
        if stack:
            raise MX3Error(f"line {stack[-1][2]}: unclosed {{")
        self.lines = lines
        return "\n".join(s for s, _ in lines) + "\n", [ln for _, ln in lines]


def translate(src, api_names, params=()):
    tr = Translator(api_names, params)
    return tr.translate(src)
