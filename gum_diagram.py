#!/usr/bin/env python3
"""
gum_diagram.py – GUM Uncertainty Tree Diagram Generator

Interactively collects a measurand definition and measurement model
as LaTeX expressions, parses them with sympy, computes partial
derivatives symbolically, and generates TikZ code for an Uncertainty
Tree Diagram (UTD) consistent with the esa-canvas-gum style.

Layout rules (matching the document conventions):
  • The root measurement model is placed at the centre.
  • Each input that has its own sub-model produces an upward branch:
      root → ∂y/∂xi (deriv_node) → sub-model block (model_block)
      The sub-model's inputs are then fanned above the model block.
  • Pure leaf inputs at the root level are placed to the right of the root.
  • Every leaf has an optional effect_node (dashed) with uncertainty sources.
  • Each input variable gets a unique colour from the palette.

Usage
-----
    python gum_diagram.py                      # interactive session
    python gum_diagram.py --example            # built-in H_s example
    python gum_diagram.py --example -o fig.tex # write output to file
    python gum_diagram.py --no-preview         # skip PNG preview

Dependencies:  sympy  (only standard dependency; no antlr4 / lark needed)
               pdflatex + gs  (for PNG preview)

LaTeX expression syntax
-----------------------
The converter handles a practical subset of LaTeX math:
  \\frac{a}{b}          →  division
  a^{n}  or  a^n        →  power (when exponent is purely numeric)
  a^{tag}               →  appended to variable name (non-numeric superscript)
  \\sqrt{x}             →  square root
  \\cdot  \\times       →  multiplication
  \\left( \\right)      →  parentheses
  Greek letters         →  safe identifier names (\\lambda → lam, etc.)
  \\mathbf{x} etc.      →  stripped to plain identifier
  a_{sub}               →  subscript becomes part of identifier name
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sympy as sp
from sympy import latex as sp_latex

# ── Colour palette (mirrors the document colour choices) ────────────────────
COLORS: List[str] = [
    "red",
    "blue!70!black",
    "purple",
    "cyan!80!black",
    "green!60!black",
    "orange!80!black",
    "magenta!70!black",
    "teal",
    "brown!70!black",
    "violet",
]

# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class InputVar:
    """One input quantity to a measurement model.

    Attributes
    ----------
    latex_name:
        The LaTeX name as the user typed it, e.g. ``\\lambda_C^{20^\\circ}``.
    sym:
        The corresponding sympy Symbol (derived automatically from latex_name).
    color:
        TikZ colour string assigned from the palette.
    submodel:
        Optional nested MeasurementModel for this input.
    effects:
        List of uncertainty-source strings shown in the dashed effect box.
    """
    latex_name: str
    sym: sp.Symbol
    color: str
    submodel: Optional["MeasurementModel"] = None
    effects: List[str] = field(default_factory=list)


@dataclass
class MeasurementModel:
    """One level of the measurement model tree.

    Attributes
    ----------
    latex_name:
        LaTeX name of the measurand, e.g. ``H_s``.
    latex_expr:
        The *right-hand side* of the model equation in LaTeX,
        e.g. ``\\left(\\frac{\\lambda_C - b}{b}\\right)^2``.
    expr:
        The sympy expression parsed from *latex_expr*.
    inputs:
        Ordered list of input variables.
    """
    latex_name: str
    latex_expr: str
    expr: sp.Expr
    inputs: List[InputVar]

    def deriv_of(self, ivar: InputVar) -> sp.Expr:
        return sp.diff(self.expr, ivar.sym)


# ── LaTeX ↔ sympy helpers ─────────────────────────────────────────────────────

def _latex_to_sym_name(latex: str) -> str:
    """Derive a safe sympy Symbol name from a LaTeX string.

    ``\\lambda_C^{20^\\circ}`` → ``lam_C20``
    Strips backslashes, braces, carets, and common LaTeX commands.
    """
    s = latex
    # Map common Greek letter commands to short names
    greek = {
        "lambda": "lam", "Lambda": "Lam",
        "theta": "theta", "Theta": "Theta",
        "phi": "phi", "Phi": "Phi",
        "sigma": "sig", "Sigma": "Sig",
        "delta": "del", "Delta": "Del",
        "alpha": "alpha", "beta": "beta", "gamma": "gamma",
        "mu": "mu", "nu": "nu", "xi": "xi", "pi": "pi",
        "rho": "rho", "tau": "tau", "omega": "omega", "Omega": "Omega",
        "eta": "eta", "kappa": "kap", "epsilon": "eps",
    }
    for cmd, rep in greek.items():
        s = re.sub(r"\\" + cmd + r"(?![A-Za-z])", rep, s)
    # Remove remaining LaTeX commands and structural characters
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\[A-Za-z]+", "", s)  # remaining commands
    s = re.sub(r"[{}^\s]", "", s)       # braces, carets, spaces
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def _find_brace_end(s: str, start: int) -> int:
    """Return index of the ``}`` matching the ``{`` at *s[start]*."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(s) - 1


def _expand_command1(s: str, cmd: str, fmt: str) -> str:
    r"""Replace ``\cmd{arg}`` with ``fmt.format(arg)`` (handles nested braces)."""
    result = []
    i = 0
    while i < len(s):
        if s[i:].startswith(cmd + "{"):
            j = i + len(cmd)
            end = _find_brace_end(s, j)
            arg = s[j + 1:end]
            result.append(fmt.format(arg))
            i = end + 1
        else:
            result.append(s[i])
            i += 1
    return "".join(result)


def _expand_command2(s: str, cmd: str, fmt: str) -> str:
    r"""Replace ``\cmd{arg1}{arg2}`` with ``fmt.format(arg1, arg2)``."""
    result = []
    i = 0
    while i < len(s):
        if s[i:].startswith(cmd + "{"):
            j = i + len(cmd)
            end1 = _find_brace_end(s, j)
            arg1 = s[j + 1:end1]
            k = end1 + 1
            while k < len(s) and s[k] == " ":
                k += 1
            if k < len(s) and s[k] == "{":
                end2 = _find_brace_end(s, k)
                arg2 = s[k + 1:end2]
                result.append(fmt.format(arg1, arg2))
                i = end2 + 1
            else:
                result.append(s[i])
                i += 1
        else:
            result.append(s[i])
            i += 1
    return "".join(result)


# Greek letter substitutions (longer names first to avoid prefix clashes)
_GREEK: List[Tuple[str, str]] = sorted([
    ("lambda", "lam"), ("Lambda", "Lam"),
    ("varphi", "phi"), ("phi", "phi"), ("Phi", "Phi"),
    ("theta", "theta"), ("Theta", "Theta"),
    ("sigma", "sig"), ("Sigma", "Sig"),
    ("delta", "del_"), ("Delta", "Del"),
    ("alpha", "alpha"), ("beta", "beta"),
    ("gamma", "gamma"), ("Gamma", "Gam"),
    ("varepsilon", "eps"), ("epsilon", "eps"),
    ("omega", "omega"), ("Omega", "Omega"),
    ("kappa", "kap"), ("eta", "eta"),
    ("mu", "mu"), ("nu", "nu"), ("xi", "xi"), ("Xi", "Xi"),
    ("pi", "pi_"), ("rho", "rho"), ("tau", "tau"),
    ("zeta", "zeta"), ("chi", "chi"),
    ("psi", "psi"), ("Psi", "Psi"),
    ("upsilon", "ups"),
], key=lambda t: -len(t[0]))


def _latex_to_sympy_str(latex: str) -> str:
    r"""Convert a LaTeX math expression to a Python string suitable for sympify.

    Handles: ``\frac``, ``\sqrt``, ``\left``/``\right``, ``\cdot``/``\times``,
    Greek letters, font commands (``\mathbf`` etc.), subscripts, and superscripts.

    Superscript rule:
      * Purely numeric/arithmetic content (digits, ``+-./``) → exponentiation.
      * All other content (letters, ``\circ``, etc.)        → appended to identifier.
    """
    s = latex.strip()

    # Expand \frac{num}{den} → ((num)/(den))
    s = _expand_command2(s, r"\frac", "(({0})/({1}))")

    # Expand \sqrt{arg} → sqrt(arg)
    s = _expand_command1(s, r"\sqrt", "sqrt({0})")

    # \left( → (  \right) → )  etc.
    s = re.sub(r"\\left\s*\(", "(", s)
    s = re.sub(r"\\right\s*\)", ")", s)
    s = re.sub(r"\\left\s*\[", "(", s)
    s = re.sub(r"\\right\s*\]", ")", s)
    s = re.sub(r"\\left\s*\.?|\\right\s*\.?", "", s)

    # Operators
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    s = re.sub(r"\\[,;!: ]", " ", s)   # spacing commands → space

    # Greek letters
    for cmd, rep in _GREEK:
        s = re.sub(r"\\" + cmd + r"(?![A-Za-z])", rep, s)

    # Font wrappers: \mathbf{x}, \mathrm{x}, \boldsymbol{x}, \text{x} → x
    for font_cmd in (r"\mathbf", r"\mathrm", r"\mathit",
                     r"\boldsymbol", r"\text", r"\operatorname"):
        s = _expand_command1(s, font_cmd, "{0}")

    # Subscripts: _{abc} → _abc  (keep as part of identifier)
    def _clean_sub(m: re.Match) -> str:
        content = re.sub(r"[^A-Za-z0-9]", "", m.group(1))
        return ("_" + content) if content else ""
    s = re.sub(r"_\{([^}]+)\}", _clean_sub, s)

    # Superscripts: ^{...} or ^x
    def _handle_super(m: re.Match) -> str:
        content = m.group(1)
        # Strip inner LaTeX commands and braces to test if purely numeric
        c = re.sub(r"\\[A-Za-z]+", "", content)
        c = re.sub(r"[{}]", "", c).strip()
        if re.match(r"^[0-9+\-./\s]+$", c):
            return f"**({c})"
        # Non-numeric superscript → append to identifier name
        clean = re.sub(r"[^A-Za-z0-9]", "", c)
        return clean if clean else ""

    s = re.sub(r"\^\{([^}]*)\}", _handle_super, s)
    s = re.sub(r"\^([A-Za-z0-9])", r"**\1", s)

    # Remove remaining unknown LaTeX commands and structural characters
    s = re.sub(r"\\[A-Za-z]+", "", s)
    s = re.sub(r"[{}\\]", "", s)

    # Implicit multiplication: digit→letter/( and )→letter/(
    s = re.sub(r"(\d)([A-Za-z(])", r"\1*\2", s)
    s = re.sub(r"\)([A-Za-z(])", r")*\1", s)

    s = re.sub(r"\s+", "", s)
    return s


def _parse_latex_expr(
    latex_rhs: str,
    symtable: Dict[str, sp.Symbol],
) -> Tuple[sp.Expr, Dict[str, sp.Symbol]]:
    """Parse a LaTeX RHS expression and return (sympy_expr, updated_symtable).

    Converts LaTeX to a sympy-compatible string via :func:`_latex_to_sympy_str`,
    then calls ``sympify``.  New symbols are registered in *symtable*; symbols
    already in *symtable* are reused so that the same variable across nested
    models refers to the same sympy Symbol.
    """
    sym_str = _latex_to_sympy_str(latex_rhs)
    try:
        parsed = sp.sympify(sym_str)
    except Exception as exc:
        raise ValueError(
            f"Could not parse LaTeX expression.\n"
            f"  Input:     {latex_rhs!r}\n"
            f"  Converted: {sym_str!r}\n"
            f"  Error:     {exc}"
        ) from exc

    # Unify symbols with the shared symtable
    subs: Dict[sp.Symbol, sp.Symbol] = {}
    for sym in parsed.free_symbols:
        name = str(sym)
        if name not in symtable:
            symtable[name] = sym
        elif symtable[name] is not sym:
            subs[sym] = symtable[name]
    if subs:
        parsed = parsed.subs(subs)

    return parsed, symtable


def _deriv_label(of_lat: str, wrt_lat: str) -> str:
    """Return ``\\frac{\\partial of}{\\partial wrt}`` LaTeX string."""
    return rf"\frac{{\partial {of_lat}}}{{\partial {wrt_lat}}}"


def _render_deriv(model: MeasurementModel, ivar: InputVar) -> str:
    """Compute ∂model/∂ivar symbolically and render to LaTeX.

    The rendered LaTeX uses the user-supplied LaTeX names for all
    input variables, not sympy's auto-generated identifier names.
    """
    d = model.deriv_of(ivar)
    raw = sp_latex(d)
    # Replace sympy symbol names with user LaTeX names (longest first)
    pairs = sorted(
        [(sp_latex(iv.sym), iv.latex_name) for iv in model.inputs],
        key=lambda p: len(p[0]),
        reverse=True,
    )
    for old, new in pairs:
        pattern = (r"(?<![A-Za-z0-9\\{])" + re.escape(old)
                   + r"(?![A-Za-z0-9_])")
        raw = re.sub(pattern, new.replace("\\", "\\\\"), raw)
    return raw


# ── Interactive session helpers ───────────────────────────────────────────────

def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return val if val else default


def _ask_yn(prompt: str, default: bool = False) -> bool:
    tag = "Y/n" if default else "y/N"
    try:
        val = input(f"{prompt} [{tag}]: ").strip().lower()
    except EOFError:
        return default
    return (val.startswith("y") if val else default)


def collect_model(
    latex_name: str,
    symtable: Dict[str, sp.Symbol],
    color_pool: List[str],
    depth: int = 0,
) -> MeasurementModel:
    """Recursively collect a measurement model via interactive prompts.

    The user types the model equation as a LaTeX RHS.  The expression is
    parsed with sympy; partial derivatives are computed automatically.
    For each detected input variable the user may optionally supply a
    sub-model or list uncertainty sources.
    """
    ind = "  " * depth
    print(f"\n{ind}── Model for  {latex_name}  ──")
    print(f"{ind}   Enter only the right-hand side in LaTeX.")
    print(rf"{ind}   Example:  \left(\frac{{\lambda_C - b}}{{b}}\right)^2")

    while True:
        latex_rhs = _ask(f"{ind}  LaTeX RHS")
        try:
            expr, symtable = _parse_latex_expr(latex_rhs, symtable)
            break
        except Exception as exc:
            print(f"{ind}  ✗ Parse error: {exc}. Please try again.")

    free_syms = sorted(expr.free_symbols, key=str)
    if free_syms:
        print(f"{ind}  Detected symbols: "
              f"{', '.join(str(s) for s in free_syms)}")
    else:
        print(f"{ind}  (No free symbols detected)")

    inputs: List[InputVar] = []
    for sym in free_syms:
        sym_str = str(sym)
        print(f"\n{ind}  ─ Input  '{sym_str}'  ─")
        latex = _ask(f"{ind}    LaTeX name", sym_str)
        color = color_pool.pop(0) if color_pool else "black"
        print(f"{ind}    Assigned colour: {color}")

        ivar = InputVar(latex_name=latex, sym=sym, color=color)

        if _ask_yn(f"{ind}    Does '{sym_str}' have a sub-model?"):
            ivar.submodel = collect_model(
                latex, dict(symtable), list(color_pool), depth + 1
            )
        else:
            raw = _ask(
                f"{ind}    Uncertainty sources (comma-separated, or blank)", ""
            )
            if raw:
                ivar.effects = [e.strip() for e in raw.split(",")]

        inputs.append(ivar)

    return MeasurementModel(
        latex_name=latex_name,
        latex_expr=latex_rhs,
        expr=expr,
        inputs=inputs,
    )


# ── TikZ code generation ─────────────────────────────────────────────────────

def _tikz_id(s: str) -> str:
    """Turn an arbitrary string into a safe TikZ node identifier."""
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def _fan_pos(idx: int, n: int, anchor: str,
             v_cm: float = 2.2, h_step: float = 2.8) -> str:
    """TikZ positioning string for a fan layout above *anchor*."""
    if n == 1:
        return f"above={v_cm}cm of {anchor}"
    total = (n - 1) * h_step
    offset = idx * h_step - total / 2.0
    if abs(offset) < 0.05:
        return f"above={v_cm}cm of {anchor}"
    if offset < 0:
        return f"above left={v_cm}cm and {abs(offset):.1f}cm of {anchor}"
    return f"above right={v_cm}cm and {offset:.1f}cm of {anchor}"


class _TikZ:
    """Thin helper that accumulates TikZ source lines."""

    def __init__(self) -> None:
        self._lines: List[str] = []

    def raw(self, s: str) -> None:
        self._lines.append(s)

    def blank(self) -> None:
        self._lines.append("")

    def comment(self, s: str, ind: int = 2) -> None:
        self._lines.append("\t" * ind + f"% {s}")

    def math_node(self, nid: str, style: str, math: str,
                  pos: str = "", extra: str = "", ind: int = 2) -> None:
        p = f", {pos}" if pos else ""
        e = f", {extra}" if extra else ""
        self._lines.append(
            "\t" * ind + rf"\node [{style}{e}{p}] ({nid}) {{${math}$}};"
        )

    def text_node(self, nid: str, style: str, text: str,
                  pos: str = "", extra: str = "", ind: int = 2) -> None:
        p = f", {pos}" if pos else ""
        e = f", {extra}" if extra else ""
        self._lines.append(
            "\t" * ind + rf"\node [{style}{e}{p}] ({nid}) {{{text}}};"
        )

    def edge(self, src: str, dst: str, style: str, ind: int = 2) -> None:
        self._lines.append(
            "\t" * ind + rf"\draw [{style}] ({src}) -- ({dst});"
        )

    def get(self) -> str:
        return "\n".join(self._lines)


class _Emitter:
    """Walks the MeasurementModel tree and emits TikZ source."""

    def __init__(self, tikz: _TikZ) -> None:
        self.t = tikz
        self._counters: Dict[str, int] = {}

    def _uid(self, base: str) -> str:
        """Return a unique TikZ node ID derived from *base*."""
        n = self._counters.get(base, 0)
        self._counters[base] = n + 1
        return base if n == 0 else f"{base}{n + 1}"

    # ── model / root block ───────────────────────────────────────────────────

    def _model_node(self, model: MeasurementModel, node_id: str,
                    is_root: bool, pos: str) -> None:
        style = "root_block" if is_root else "model_block"
        content = rf"{model.latex_name} = {model.latex_expr}"
        label = "ROOT" if is_root else "sub-model"
        self.t.comment(f"{label}: {model.latex_name}")
        self.t.math_node(node_id, style, content, pos=pos)
        self.t.blank()

    # ── root-level layout (up-branches + side branches) ─────────────────────

    def emit_root(self, model: MeasurementModel, root_id: str) -> None:
        self._model_node(model, root_id, is_root=True, pos="")

        up_vars = [iv for iv in model.inputs if iv.submodel is not None]
        side_vars = [iv for iv in model.inputs if iv.submodel is None]

        # Upward branches: one or more inputs with sub-models
        n_up = len(up_vars)
        for i, ivar in enumerate(up_vars):
            d_id = self._uid(f"D{_tikz_id(ivar.latex_name)}")
            m_id = self._uid(f"M{_tikz_id(ivar.latex_name)}")
            d_pos = _fan_pos(i, n_up, root_id, v_cm=2.0, h_step=3.5)
            self._up_branch(model, root_id, ivar, d_id, m_id, d_pos)

        # Side branches: leaf inputs at the root level
        prev_d = root_id
        for i, ivar in enumerate(side_vars):
            d_id = self._uid(f"D{_tikz_id(ivar.latex_name)}S")
            leaf_id = self._uid(f"U{_tikz_id(ivar.latex_name)}")
            if i == 0:
                d_pos = f"right=1.0cm of {root_id}"
            else:
                d_pos = f"below=0.8cm of {prev_d}"
            self._side_leaf(model, root_id, ivar, d_id, leaf_id, d_pos)
            prev_d = d_id

    def _up_branch(self, parent: MeasurementModel, parent_id: str,
                   ivar: InputVar, d_id: str, m_id: str, d_pos: str) -> None:
        """Emit  parent → deriv_node → sub-model block → sub-model inputs."""
        color = ivar.color
        deriv_lat = _deriv_label(parent.latex_name, ivar.latex_name)
        self.t.comment(f"up-branch: ∂{parent.latex_name}/∂{ivar.latex_name}")
        self.t.math_node(d_id, "deriv_node", deriv_lat, pos=d_pos)
        self.t.edge(parent_id, d_id, f"connection, {color}")
        self.t.blank()
        # Sub-model block above the derivative
        self._model_node(ivar.submodel, m_id, is_root=False,
                         pos=f"above=of {d_id}")
        self.t.edge(d_id, m_id, f"connection, {color}")
        # Fan the sub-model's own inputs above it
        self._fan_inputs(ivar.submodel, m_id)

    # ── fan layout: all inputs of a sub-model fanned above it ───────────────

    def _fan_inputs(self, model: MeasurementModel, model_id: str) -> None:
        n = len(model.inputs)
        for i, ivar in enumerate(model.inputs):
            pos = _fan_pos(i, n, model_id, v_cm=2.2, h_step=2.8)
            if ivar.submodel is not None:
                self._fan_submodel(model, model_id, ivar, pos)
            else:
                self._fan_leaf(model, model_id, ivar, pos)

    def _fan_submodel(self, parent: MeasurementModel, parent_id: str,
                      ivar: InputVar, pos: str) -> None:
        color = ivar.color
        d_id = self._uid(f"D{_tikz_id(ivar.latex_name)}")
        m_id = self._uid(f"M{_tikz_id(ivar.latex_name)}")
        deriv_lat = _deriv_label(parent.latex_name, ivar.latex_name)
        self.t.comment(f"nested sub-model for {ivar.latex_name}")
        self.t.math_node(d_id, "deriv_node", deriv_lat, pos=pos)
        self.t.edge(parent_id, d_id, f"connection, {color}")
        self.t.blank()
        self._model_node(ivar.submodel, m_id, is_root=False,
                         pos=f"above=of {d_id}")
        self.t.edge(d_id, m_id, f"connection, {color}")
        self._fan_inputs(ivar.submodel, m_id)

    def _fan_leaf(self, parent: MeasurementModel, parent_id: str,
                  ivar: InputVar, pos: str) -> None:
        color = ivar.color
        d_id = self._uid(f"D{_tikz_id(ivar.latex_name)}")
        leaf_id = self._uid(f"U{_tikz_id(ivar.latex_name)}")
        deriv_lat = _deriv_label(parent.latex_name, ivar.latex_name)
        self.t.comment(f"leaf: u({ivar.latex_name})")
        self.t.math_node(d_id, "deriv_node", deriv_lat, pos=pos)
        self.t.math_node(
            leaf_id, "leaf_node", rf"u({ivar.latex_name})",
            pos=f"above=of {d_id}",
            extra=f"draw={color}, text={color}",
        )
        self.t.edge(parent_id, d_id, f"connection, {color}")
        self.t.edge(d_id, leaf_id, f"connection, {color}")
        self._effect_node(ivar, leaf_id)
        self.t.blank()

    # ── side leaf (root-level leaf input, placed to the right) ───────────────

    def _side_leaf(self, parent: MeasurementModel, parent_id: str,
                   ivar: InputVar, d_id: str, leaf_id: str, d_pos: str) -> None:
        color = ivar.color
        deriv_lat = _deriv_label(parent.latex_name, ivar.latex_name)
        self.t.comment(f"side leaf: u({ivar.latex_name})")
        self.t.math_node(d_id, "deriv_node", deriv_lat, pos=d_pos)
        self.t.math_node(
            leaf_id, "leaf_node", rf"u({ivar.latex_name})",
            pos=f"right=0.5cm of {d_id}",
            extra=f"draw={color}, text={color}",
        )
        self.t.edge(parent_id, d_id, f"connection, {color}")
        self.t.edge(d_id, leaf_id, f"connection, {color}")
        self._effect_node(ivar, leaf_id)
        self.t.blank()

    # ── effect node (dashed box with uncertainty sources) ────────────────────

    def _effect_node(self, ivar: InputVar, leaf_id: str) -> None:
        if not ivar.effects:
            return
        eff_id = self._uid(f"EFF{_tikz_id(ivar.latex_name)}")
        eff_text = r" \\ ".join(ivar.effects)
        self.t.text_node(
            eff_id, "effect_node", eff_text,
            pos=f"above=0.4cm of {leaf_id}",
            extra=ivar.color,
        )
        self.t.edge(eff_id, leaf_id, f"connection, {ivar.color}, dashed")


def build_tikz(root: MeasurementModel, label: str = "") -> str:
    """Return a complete LaTeX figure environment with the TikZ UTD.

    Parameters
    ----------
    root:
        The root measurement model.
    label:
        LaTeX ``\\label`` key used inside the figure (without the
        ``fig:`` prefix).  Defaults to ``utd_<ROOTSYM>``.
    """
    if not label:
        label = f"utd_{_tikz_id(root.latex_name).lower()}"
    t = _TikZ()
    t.raw(r"\begin{figure}[H]")
    t.raw(r"  \centering")
    t.raw(r"  \resizebox{\textwidth}{!}{%")
    t.raw(r"  \begin{tikzpicture}[")
    t.raw(r"    node distance=0.7cm and 0.6cm,")
    t.raw(r"    connection/.style={draw, thick},")
    t.raw(r"    root_block/.style={draw, rectangle, inner sep=10pt,"
          r" font=\Large\bfseries, align=center},")
    t.raw(r"    model_block/.style={draw, rectangle, inner sep=8pt,"
          r" font=\large\bfseries, align=center},")
    t.raw(r"    deriv_node/.style={draw, rectangle, rounded corners=5pt,"
          r" inner sep=5pt, font=\normalsize, align=center},")
    t.raw(r"    leaf_node/.style={draw, rectangle, inner sep=5pt,"
          r" font=\small, align=center, text width=1.3cm},")
    t.raw(r"    effect_node/.style={draw, dashed, font=\footnotesize\itshape,"
          r" align=center, text width=2.3cm, inner sep=3pt}")
    t.raw(r"    ]")

    root_id = _tikz_id(root.latex_name) + "ROOT"
    emitter = _Emitter(t)
    emitter.emit_root(root, root_id)

    t.raw(r"  \end{tikzpicture}%")
    t.raw(r"  }")
    t.raw(rf"  \caption{{Uncertainty Tree Diagram for ${root.latex_name}$.}}")
    t.raw(rf"  \label{{fig:{label}}}")
    t.raw(r"\end{figure}")
    return t.get()


# ── Built-in example (H_s from the document) ─────────────────────────────────

def _builtin_example() -> MeasurementModel:
    """Reconstruct the H_s uncertainty tree diagram from the document."""
    R, theta, phi0, lam_C = sp.symbols("R theta phi0 lam_C")

    R_iv = InputVar(r"R", R, "purple",
                    effects=["Geolocation", "Orbital Fitting",
                             "Antenna mis-pointing"])
    theta_iv = InputVar(r"\theta", theta, "cyan!80!black",
                        effects=["Geolocation", "Orbital Fitting",
                                 "Antenna mis-pointing"])
    lam_C_iv = InputVar(r"\lambda_C", lam_C, "red",
                        effects=["Bright Target Removal", "FFT", "CCS",
                                 "Range-avg CCS", "IFFT", "Gaussian fit"])
    phi0_iv = InputVar(r"\phi_0", phi0, "green!60!black",
                       effects=["NWP modelling"])

    lam_C20_expr = lam_C * R / (theta * phi0)
    lam_C20_model = MeasurementModel(
        latex_name=r"\lambda_C^{20^\circ}",
        latex_expr=r"\frac{\lambda_C \cdot R}{\theta \cdot \phi_0}",
        expr=lam_C20_expr,
        inputs=[R_iv, theta_iv, lam_C_iv, phi0_iv],
    )

    lam_C20, b = sp.symbols("lam_C20 b")
    hs_expr = ((lam_C20 - b) / b) ** 2

    lam_C20_iv = InputVar(r"\lambda_C^{20^\circ}", lam_C20, "red",
                          submodel=lam_C20_model)
    b_iv = InputVar(r"\mathbf{b}", b, "blue!70!black",
                    effects=["Linear Fitting"])

    return MeasurementModel(
        latex_name=r"H_s",
        latex_expr=r"\left(\frac{\lambda_C^{20^\circ} - \mathbf{b}}{\mathbf{b}}\right)^{2}",
        expr=hs_expr,
        inputs=[lam_C20_iv, b_iv],
    )


def _label_to_filename(label: str) -> str:
    """Convert a figure label to a safe .tex filename.

    ``fig:swh_utd`` → ``fig_swh_utd.tex``
    Colons become underscores; other non-alphanumeric/dash characters are
    stripped; the result is lower-cased.
    """
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", label).strip("_")
    safe = re.sub(r"_+", "_", safe)
    return safe.lower() + ".tex"


# ── PNG rendering ─────────────────────────────────────────────────────────────

# Minimal LaTeX wrapper that compiles a bare \begin{figure}…\end{figure} snippet.
_LATEX_WRAPPER = r"""
\documentclass[border=6pt]{{standalone}}
\usepackage{{tikz}}
\usetikzlibrary{{positioning, calc}}
\usepackage{{amsmath, amssymb}}
\usepackage{{xcolor}}
\usepackage{{float}}
\begin{{document}}
{body}
\end{{document}}
""".lstrip()


def render_png(tex_path: str, png_path: str, dpi: int = 150) -> bool:
    """Compile *tex_path* with pdflatex and convert to *png_path*.

    The .tex file is expected to contain a bare ``\\begin{figure}…\\end{figure}``
    block (as produced by :func:`build_tikz`).  It is wrapped in a minimal
    standalone document, compiled with ``pdflatex``, and converted to PNG via
    Ghostscript.

    Returns ``True`` on success, ``False`` if any step fails (with a warning
    printed to stderr).
    """
    import subprocess
    import tempfile
    import os

    # Read the figure snippet
    try:
        body = Path(tex_path).read_text()
    except OSError as exc:
        print(f"  ✗  Cannot read {tex_path}: {exc}", file=sys.stderr)
        return False

    # Strip \begin{figure}/\end{figure} and caption/label lines — standalone
    # doesn't want those, just the tikzpicture content.
    body_inner = re.sub(
        r"\\begin\{figure\}[^\n]*\n?", "", body
    )
    body_inner = re.sub(r"\\end\{figure\}", "", body_inner)
    body_inner = re.sub(r"\s*\\caption\{[^}]*\}\n?", "", body_inner)
    body_inner = re.sub(r"\s*\\label\{[^}]*\}\n?", "", body_inner)
    body_inner = re.sub(r"\\centering\n?", "", body_inner)

    wrapper = _LATEX_WRAPPER.format(body=body_inner.strip())

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "fig.tex")
        pdf = os.path.join(td, "fig.pdf")
        Path(src).write_text(wrapper)

        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", td, src],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not os.path.exists(pdf):
            print("  ✗  pdflatex failed:", file=sys.stderr)
            for line in result.stdout.splitlines()[-20:]:
                print("     " + line, file=sys.stderr)
            return False

        # Convert PDF → PNG using Ghostscript
        gs_result = subprocess.run(
            [
                "gs", "-dBATCH", "-dNOPAUSE", "-dQUIET",
                "-sDEVICE=png16m", f"-r{dpi}",
                f"-sOutputFile={png_path}", pdf,
            ],
            capture_output=True, text=True,
        )
        if gs_result.returncode != 0 or not os.path.exists(png_path):
            print(f"  ✗  Ghostscript failed: {gs_result.stderr.strip()}",
                  file=sys.stderr)
            return False

    return True


def _open_image(png_path: str) -> None:
    """Open *png_path* with the system default viewer (best-effort)."""
    import subprocess
    import shutil

    viewers = ["eog", "feh", "display", "xdg-open", "open"]
    for viewer in viewers:
        if shutil.which(viewer):
            subprocess.Popen(
                [viewer, png_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
    print(f"  ℹ  No image viewer found; open {png_path} manually.",
          file=sys.stderr)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate TikZ Uncertainty Tree Diagrams for GUM analyses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--example", action="store_true",
        help="Use the built-in H_s example instead of interactive input",
    )
    ap.add_argument(
        "-o", "--output", default=None, metavar="FILE",
        help=(
            "Output .tex file.  Defaults to <label>.tex in the current "
            "directory, where <label> is derived from the figure label."
        ),
    )
    ap.add_argument(
        "--no-preview", action="store_true",
        help="Skip PNG rendering and display (just write the .tex file)",
    )
    ap.add_argument(
        "--dpi", type=int, default=150, metavar="N",
        help="Resolution for the PNG preview (default: 150)",
    )
    args = ap.parse_args()

    if args.example:
        model = _builtin_example()
        label = "swh_utd"
    else:
        print("╔══════════════════════════════════════════════════════╗")
        print("║  GUM Uncertainty Tree Diagram Generator              ║")
        print("╚══════════════════════════════════════════════════════╝")
        print()
        print("Step 1: Measurand")
        lat_name = _ask(r"  LaTeX name  (e.g.  H_s,  \sigma^0)")
        default_label = f"utd_{_latex_to_sym_name(lat_name).lower()}"
        label = _ask(
            r"  Figure label (\label{fig:<…>}, without 'fig:')",
            default_label,
        )
        print()
        print("Step 2: Measurement model")
        symtable: Dict[str, sp.Symbol] = {}
        color_pool = list(COLORS)
        model = collect_model(lat_name, symtable, color_pool)

    tikz_code = build_tikz(model, label=label)

    out_path = args.output if args.output else _label_to_filename(label)
    with open(out_path, "w") as fh:
        fh.write(tikz_code + "\n")
    print(f"\n✓  TikZ code written to  {out_path}")
    print(f"   Include in LaTeX with:  \\input{{{out_path}}}")
    print(f"   Reference with:         \\ref{{fig:{label}}}")

    if not args.no_preview:
        png_path = out_path.replace(".tex", ".png")
        print(f"\n   Rendering PNG preview … ", end="", flush=True)
        ok = render_png(out_path, png_path, dpi=args.dpi)
        if ok:
            print(f"saved to  {png_path}")
            _open_image(png_path)
        else:
            print("failed (use --no-preview to skip)")


if __name__ == "__main__":
    main()
