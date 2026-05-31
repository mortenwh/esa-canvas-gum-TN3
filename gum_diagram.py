#!/usr/bin/env python3
"""
gum_diagram.py – GUM Uncertainty Tree Diagram Generator

Interactively collects a measurand definition and measurement model
as LaTeX expressions, parses them with sympy, computes partial
derivatives symbolically, and generates TikZ code for an Uncertainty
Tree Diagram (UTD) consistent with the esa-canvas-gum style.

Layout rules (matching the document conventions):
  • The root measurement model is placed at the centre.
  • Root-level inputs radiate outward at angles distributed in an arc.
    Each input is allocated angular width proportional to its leaf count,
    so the arc grows with the number of branches:
      ≤ 6 leaves  → top arc only  (≤ 180°)
      ~12 leaves  → extends to sides (270°)
      many leaves → nearly full circle (capped at 330°)
  • The sub-tree of each branch grows outward in that branch's angle,
    with children spread perpendicularly.
  • Every leaf has an optional effect_node (dashed) with uncertainty sources.
  • Each input variable gets a unique colour from the palette.

Usage
-----
    python gum_diagram.py                      # interactive session
    python gum_diagram.py --example            # built-in H_s example
    python gum_diagram.py --example -o fig.tex # write output to file
    python gum_diagram.py --no-preview         # skip PNG preview

Output
------
The tool writes a self-contained LaTeX figure environment to <label>.tex
(or the file given with -o).  To include the figure in your document:

  1. Copy the .tex file next to your main .tex source (or to a subfolder).

  2. Make sure the following packages are loaded in your preamble::

       \\usepackage{tikz}
       \\usetikzlibrary{positioning,calc}
       \\usepackage{amsmath,amssymb,xcolor,adjustbox}

  3. Include the figure at the desired location::

       \\input{<label>.tex}

     The figure is wrapped in a ``figure`` environment and will float.
     Refer to it in the text with::

       \\ref{fig:<label>}   or   \\autoref{fig:<label>}

     Sub-models marked as separate figures appear in the parent as a
     model-equation block with a cross-reference note.  Run the tool
     again to generate each sub-model figure separately.
  4. (Optional) To keep all figures together in the appendix, use::

       \\usepackage{float}
       \\floatplacement{figure}{p}   % one figure per page (already the default)

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
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    separate_figure: bool = False   # sub-model traced in a separate figure
    separate_label: str = ""        # \ref label for the cross-reference (without 'fig:')
    branch_offset: Tuple[float, float] = (0.0, 0.0)  # auto/manual translation (cm)


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
    ``\\Delta\\varpi_{\\rm dc}`` → ``Delvarpi_dc``
    Strips backslashes, braces, carets, and common LaTeX commands.
    """
    s = latex
    # Font wrappers must come first (before single-pass command substitution)
    for fc in (r"\\mathbf", r"\\mathrm", r"\\mathit",
               r"\\boldsymbol", r"\\text", r"\\operatorname"):
        s = re.sub(fc + r"\{([^}]+)\}", r"\1", s)
    # Single-pass Greek/command substitution (avoids serial-loop prefix clashes)
    _sym_greek = {cmd: rep for cmd, rep in _GREEK}
    # Bare font-switch commands (\rm, \bf …) and spacing → strip
    _sym_greek.update({"rm": "", "bf": "", "it": "",
                       "sf": "", "tt": "", "cal": "",
                       "circ": "", "deg": ""})

    def _sub(m: re.Match) -> str:
        return _sym_greek.get(m.group(1), m.group(1))

    s = re.sub(r"\\([A-Za-z]+)(?![A-Za-z])", _sub, s)
    s = re.sub(r"\\[A-Za-z]+", "", s)  # any remaining commands
    s = re.sub(r"[{}^\s]", "", s)       # braces, carets, spaces
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


_UNICODE_GREEK: Dict[str, str] = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ",
    "sigma": "σ", "varsigma": "ς", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "varphi": "ϕ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Alpha": "Α", "Beta": "Β", "Gamma": "Γ", "Delta": "Δ",
    "Epsilon": "Ε", "Zeta": "Ζ", "Eta": "Η", "Theta": "Θ",
    "Iota": "Ι", "Kappa": "Κ", "Lambda": "Λ", "Mu": "Μ",
    "Nu": "Ν", "Xi": "Ξ", "Pi": "Π", "Rho": "Ρ", "Sigma": "Σ",
    "Tau": "Τ", "Upsilon": "Υ", "Phi": "Φ", "Chi": "Χ",
    "Psi": "Ψ", "Omega": "Ω",
    "circ": "°", "deg": "°",
    "infty": "∞", "nabla": "∇", "partial": "∂",
    "cdot": "·", "times": "×",
}


def _latex_to_unicode(s: str) -> str:
    r"""Convert a LaTeX symbol token to a Unicode terminal approximation.

    ``\\Delta\\varpi_{\\rm dc}`` → ``Δϖ_dc``
    ``\\lambda_C^{20^\\circ}``   → ``λ_C^20°``
    ``\\mathbf{t}``              → ``t``
    """
    # Strip \left / \right wrappers
    s = re.sub(r"\\(?:left|right)\s*", "", s)

    # Strip font-wrapper commands: \mathbf{x} → x  (iterate for nesting)
    _fw_re = re.compile(
        r"\\(?:mathbf|mathrm|mathit|mathsf|mathtt|mathcal|boldsymbol|text)\{"
    )
    while True:
        m = _fw_re.search(s)
        if not m:
            break
        brace_i = m.end() - 1
        # find matching closing brace manually
        depth, i = 0, brace_i
        while i < len(s):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        inner = s[brace_i + 1:i]
        s = s[:m.start()] + inner + s[i + 1:]

    # Bare font-switch commands (\rm, \bf …) → stripped
    s = re.sub(r"\\(?:rm|bf|it|sf|tt|cal)\b\s*", "", s)

    # Subscripts _{content} → unicode subscript digits or _text
    def _sub_repl(m: re.Match) -> str:
        inner = _latex_to_unicode(m.group(1))
        if re.match(r"^[0-9]+$", inner):
            return "".join(chr(0x2080 + int(c)) for c in inner)
        return "_" + inner

    s = re.sub(r"_\{([^{}]*)\}", _sub_repl, s)
    s = re.sub(r"_([A-Za-z0-9])", lambda m: "_" + m.group(1), s)

    # Superscripts ^{content} → ° if purely decorative, else ^text
    def _sup_repl(m: re.Match) -> str:
        inner = _latex_to_unicode(m.group(1))
        if inner.strip() == "°":
            return "°"
        return ("^" + inner) if inner else ""

    s = re.sub(r"\^\{([^{}]*)\}", _sup_repl, s)
    s = re.sub(r"\^([A-Za-z0-9])", lambda m: "^" + m.group(1), s)

    # Single-pass Greek / symbol substitution; unknown commands stripped
    def _greek_cb(m: re.Match) -> str:
        return _UNICODE_GREEK.get(m.group(1), "")

    s = re.sub(r"\\([A-Za-z]+)", _greek_cb, s)

    # Remove orphaned carets (not followed by a letter/digit) left after command
    # substitution, e.g. the ^ before \circ in 20^\circ → 20^° → 20°
    s = re.sub(r"\^(?![A-Za-z0-9])", "", s)

    # Remove leftover braces
    s = re.sub(r"[{}]", "", s)
    return s.strip()


def _sym_display(latex_token: str) -> str:
    """Return a terminal label: ``unicode  (latex)`` or just ``unicode``.

    If the Unicode approximation equals the plain token (e.g. 'P'), the
    parenthesised LaTeX part is omitted to keep the output clean.
    """
    uni = _latex_to_unicode(latex_token)
    if uni == latex_token:
        return uni
    return f"{uni}  ({latex_token})"


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
    ("varphi", "varph"), ("vartheta", "vartheta"), ("varpi", "varpi"),
    ("varrho", "varrho"), ("varsigma", "varsi"), ("varepsilon", "vareps"),
    ("phi", "phi"), ("Phi", "Phi"),
    ("theta", "theta"), ("Theta", "Theta"),
    ("sigma", "sig"), ("Sigma", "Sig"),
    ("delta", "del_"), ("Delta", "Del"),
    ("alpha", "alpha"), ("beta", "beta"),
    ("gamma", "gamma"), ("Gamma", "Gam"),
    ("epsilon", "eps"),
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

    # Greek letters — single regex pass so adjacent commands don't interfere
    # (serial substitution turns \Delta\varpi → \Deltavarpi, breaking \Delta)
    _gd = {cmd: rep for cmd, rep in _GREEK}

    def _greek_sub(m: re.Match) -> str:
        return _gd.get(m.group(1), m.group(0))

    s = re.sub(r"\\([A-Za-z]+)(?![A-Za-z])", _greek_sub, s)

    # Font wrappers: \mathbf{x}, \mathrm{x}, \boldsymbol{x}, \text{x} → x
    for font_cmd in (r"\mathbf", r"\mathrm", r"\mathit",
                     r"\boldsymbol", r"\text", r"\operatorname"):
        s = _expand_command1(s, font_cmd, "{0}")

    # Subscripts: _{abc} → _abc  (keep as part of identifier)
    def _clean_sub(m: re.Match) -> str:
        content = m.group(1)
        # Strip any LaTeX commands inside the subscript (e.g. \rm, \mathrm)
        content = re.sub(r"\\[A-Za-z]+\s*", "", content)
        content = re.sub(r"[^A-Za-z0-9]", "", content)
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

    # Clean up empty/duplicate arguments left by removed commands
    # e.g. g(P,,t) → g(P,t)  or  g(,t) → g(t)
    s = re.sub(r",\s*,", ",", s)           # double commas
    s = re.sub(r"\(\s*,", "(", s)          # leading comma after (
    s = re.sub(r",\s*\)", ")", s)           # trailing comma before )
    s = re.sub(r",\s*,", ",", s)           # second pass after above fixes

    # Implicit multiplication: digit→letter/( and )→letter/(
    s = re.sub(r"(\d)([A-Za-z(])", r"\1*\2", s)
    s = re.sub(r"\)([A-Za-z(])", r")*\1", s)

    s = re.sub(r"\s+", "", s)
    return s


def _extract_latex_vars(latex: str) -> Dict[str, str]:
    """Extract variable-like tokens from a LaTeX expression.

    Returns a dict mapping ``sympy_name → latex_token`` for each
    variable-like token found in *latex*.  This lets the interactive
    session show the original LaTeX symbol rather than the internal
    sympy identifier.

    A variable token starts with a ``\\cmd`` or plain letter, may be
    prefixed by a modifier command (``\\Delta``, ``\\hat`` etc.), and
    may be followed by subscripts ``_{...}`` and superscripts ``^{...}``.
    Font wrappers (``\\mathbf{arg}``) are also recognised.

    Structural/operator commands (``\\frac``, ``\\sqrt``, …) are skipped.
    """
    _structural = {
        "frac", "sqrt", "left", "right", "cdot", "times",
        "begin", "end", "sum", "prod", "int", "oint", "lim",
        "partial", "infty", "pm", "mp", "le", "ge", "neq",
        "rm", "bf", "it", "sf", "tt", "cal",
        "over", "under", "overbrace", "underbrace",
    }
    _font_cmds = {"mathbf", "mathrm", "mathit", "boldsymbol",
                  "text", "operatorname"}
    # Modifier prefixes that attach to the next atom (e.g. \Delta\varpi)
    _modifiers = {"hat", "tilde", "bar", "vec", "dot", "ddot",
                  "Delta", "delta", "nabla", "partial"}

    result: Dict[str, str] = {}

    def _read_cmd(s: str, pos: int) -> Tuple[str, int]:
        """Read a ``\\cmd`` starting at s[pos]=='\\'. Return (cmd, end_pos)."""
        j = pos + 1
        while j < len(s) and s[j].isalpha():
            j += 1
        return s[pos + 1:j], j

    def _opt_subsup(s: str, pos: int) -> Tuple[str, int]:
        """Consume optional ``_{...}``/``^{...}`` (or ``_x``/``^x``)."""
        extra = ""
        i = pos
        while i < len(s):
            if s[i] in ("_", "^") and i + 1 < len(s):
                if s[i + 1] == "{":
                    end = _find_brace_end(s, i + 1)  # handles nested braces
                    extra += s[i:end + 1]
                    i = end + 1
                else:
                    extra += s[i:i + 2]
                    i += 2
            else:
                break
        return extra, i

    def _read_atom(s: str, pos: int):
        """Read one variable atom (cmd, font-wrapped, or plain letter).

        Returns (token_str, end_pos) or (None, pos) if not a variable atom.
        """
        if pos >= len(s):
            return None, pos
        if s[pos] == "\\":
            cmd, j = _read_cmd(s, pos)
            if cmd in _font_cmds and j < len(s) and s[j] == "{":
                end = _find_brace_end(s, j)
                token = s[pos:end + 1]
                return token, end + 1
            elif cmd and cmd not in _structural:
                return s[pos:j], j
            else:
                return None, pos
        elif s[pos].isalpha():
            j = pos
            while j < len(s) and s[j].isalpha():
                j += 1
            return s[pos:j], j
        return None, pos

    i = 0
    while i < len(latex):
        # Skip whitespace and non-variable characters
        if latex[i] in " \t\n,;()[]{}+=-*/^_|<>!&":
            i += 1
            continue

        if latex[i] == "\\":
            cmd, j = _read_cmd(latex, i)
            if cmd in _structural:
                i = j
                continue

            # Check for modifier prefix (e.g. \Delta before \varpi)
            if cmd in _modifiers:
                prefix_token = latex[i:j]
                # Try to read the next atom
                # skip whitespace
                k = j
                while k < len(latex) and latex[k] == " ":
                    k += 1
                atom, k2 = _read_atom(latex, k)
                if atom:
                    # Combined token: \Delta\varpi_{\rm dc}
                    full = prefix_token + atom
                    extra, end = _opt_subsup(latex, k2)
                    full += extra
                    sym_name = _latex_to_sym_name(full)
                    if sym_name:
                        result[sym_name] = full
                    i = end
                    continue
                # Modifier without following atom — treat as standalone
                extra, end = _opt_subsup(latex, j)
                full = prefix_token + extra
                sym_name = _latex_to_sym_name(full)
                if sym_name:
                    result[sym_name] = full
                i = end
                continue

            if cmd in _font_cmds and j < len(latex) and latex[j] == "{":
                end = _find_brace_end(latex, j)
                token = latex[i:end + 1]
                extra, end2 = _opt_subsup(latex, end + 1)
                token += extra
                sym_name = _latex_to_sym_name(token)
                if sym_name:
                    result[sym_name] = token
                i = end2
                continue

            # Regular \cmd (Greek letter etc.)
            token = latex[i:j]
            extra, end = _opt_subsup(latex, j)
            token += extra
            sym_name = _latex_to_sym_name(token)
            if sym_name:
                result[sym_name] = token
            i = end

        elif latex[i].isalpha():
            j = i
            while j < len(latex) and latex[j].isalpha():
                j += 1
            word = latex[i:j]
            token = word
            extra, end = _opt_subsup(latex, j)
            token += extra
            sym_name = _latex_to_sym_name(token)
            if sym_name:
                result[sym_name] = token
            i = end

        else:
            i += 1

    return result


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

    # Build sympify locals: start from symtable, then override sympy names
    # that would silently swallow user variables:
    #   Single-letter: I=ImaginaryUnit, E=Euler's number, S=singleton,
    #                  N=numerical evaluator, C=category, O=Order, Q=assumptions
    #   Greek-derived: gamma/beta/zeta → sympy special functions (no free_symbols);
    #                  these are common physics variables, not special functions.
    # We do NOT override 'pi' (users writing \pi almost always mean the constant).
    _RESERVED = {"I", "E", "S", "N", "C", "O", "Q", "gamma", "beta", "zeta"}
    sym_locals: Dict[str, Any] = dict(symtable)
    for name in _RESERVED:
        if name in sym_locals:
            continue
        # Check whether the name appears as a function call (name followed by ()
        if re.search(rf"(?<![A-Za-z_]){re.escape(name)}\s*\(", sym_str):
            sym_locals[name] = sp.Function(name)
        elif re.search(rf"(?<![A-Za-z_]){re.escape(name)}(?![A-Za-z_0-9(])",
                       sym_str):
            sym_locals[name] = sp.Symbol(name)

    try:
        parsed = sp.sympify(sym_str, locals=sym_locals)
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
    shallow: bool = False,
) -> MeasurementModel:
    """Recursively collect a measurement model via interactive prompts.

    The user types the model equation as a LaTeX RHS.  The expression is
    parsed with sympy; partial derivatives are computed automatically.
    For each detected input variable the user may optionally supply a
    sub-model or list uncertainty sources.

    Parameters
    ----------
    shallow:
        When True (used for separate-figure sub-models) only the equation
        is collected; no further questions are asked about each input.
        All inputs become plain leaves.
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
    # Build sym_name → LaTeX token map from the original expression
    latex_var_map = _extract_latex_vars(latex_rhs)

    if free_syms:
        display = ", ".join(
            _sym_display(latex_var_map.get(str(s), str(s))) for s in free_syms
        )
        print(f"{ind}  Detected symbols: {display}")
    else:
        print(f"{ind}  (No free symbols detected)")

    inputs: List[InputVar] = []
    for sym in free_syms:
        sym_str = str(sym)
        latex = latex_var_map.get(sym_str, sym_str)
        color = color_pool.pop(0) if color_pool else "black"
        ivar = InputVar(latex_name=latex, sym=sym, color=color)

        if not shallow:
            label = _sym_display(latex)
            print(f"\n{ind}  ─ Input  {label}  ─")
            print(f"{ind}    Assigned colour: {color}")

            if _ask_yn(f"{ind}    Does {label} have a sub-model?"):
                if _ask_yn(f"{ind}    Trace {label} in a separate figure?"):
                    ivar.submodel = collect_model(
                        latex, dict(symtable), list(color_pool), depth + 1,
                        shallow=True,
                    )
                    ivar.separate_figure = True
                    default_sep_label = f"utd_{_latex_to_sym_name(latex).lower()}"
                    ivar.separate_label = _ask(
                        f"{ind}    Separate figure label (without 'fig:')",
                        default_sep_label,
                    )
                else:
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


# Layout constants (all in cm)
_H_LEAF  = 1.8   # horizontal space allocated per leaf node (level 0)
_V_D0    = 1.8   # root → root-level deriv_node
_V_M0    = 1.4   # deriv_node → sub-model block
_V_D1    = 1.8   # sub-model block → nested deriv_node
_V_LEAF  = 1.4   # any deriv_node → leaf_node
_V_EFF   = 1.2   # leaf_node → effect_node
_DEPTH_SCALE = 0.85  # multiply H_LEAF and V distances by this factor per depth level
_H_LEAF_MIN  = 1.3   # minimum horizontal spacing (cm) to prevent box overlap
# Minimum distances that override depth-scaling (prevent within-branch overlaps)
_V_M0_MIN   = 1.8   # minimum deriv→model distance
_V_D1_MIN   = 1.8   # minimum model→child-deriv distance
_V_LEAF_MIN = 1.4   # minimum deriv→leaf distance
# Arc parameters for radial sub-fan (sub-model children)
_ARC_PER_LEAF_SUB = 20.0   # degrees per leaf for sub-model fan
_ARC_CAP_SUB_DEG  = 120.0  # max fan arc for any sub-model


def _leaf_count(ivar: "InputVar") -> int:
    """Total leaf nodes in the subtree rooted at *ivar* (minimum 1).

    Separate-figure sub-models count as 1 leaf regardless of their
    shallow-model size — they render as a single compact node.
    """
    if ivar.submodel is None:
        return 1
    if ivar.separate_figure:
        return 1
    return max(1, sum(_leaf_count(iv) for iv in ivar.submodel.inputs))


def _child_offsets(inputs: "List[InputVar]",
                   h_unit: float = _H_LEAF) -> "List[float]":
    """Return horizontal offsets (cm, centred at 0) for each input's subtree."""
    widths = [_leaf_count(iv) * h_unit for iv in inputs]
    total = sum(widths)
    xs: List[float] = []
    x = -total / 2.0
    for w in widths:
        xs.append(x + w / 2.0)
        x += w
    return xs


# Degrees of arc allocated per leaf node when distributing root branches.
# 30° per leaf → 6 leaves fill 180°, 12 leaves fill 360° (capped at 330°).
_ARC_PER_LEAF = 30.0
_ARC_CAP_DEG  = 330.0


def _root_angles(inputs: "List[InputVar]") -> "List[float]":
    """Return the outward angle (radians) for each root-level input.

    Branches are distributed in an arc centred at π/2 (straight up).
    Angular width is proportional to each branch's leaf count (at least 1
    per branch to prevent zero-width slices for separate-figure stubs).

    The total arc grows at ``_ARC_PER_LEAF`` degrees per total leaf,
    capped at ``_ARC_CAP_DEG``.
    """
    if not inputs:
        return []
    raw_counts = [_leaf_count(iv) for iv in inputs]
    total_leaves = sum(raw_counts)
    smoothed = [max(c, 1) for c in raw_counts]
    smoothed_total = sum(smoothed)

    arc_deg = min(_ARC_CAP_DEG, total_leaves * _ARC_PER_LEAF)
    arc_rad = math.radians(arc_deg)
    start = math.pi / 2 + arc_rad / 2          # left-most angle
    angles: List[float] = []
    cumulative = 0.0
    for s in smoothed:
        frac = s / smoothed_total
        angles.append(start - (cumulative + frac / 2) * arc_rad)
        cumulative += frac
    return angles


def _sub_angles(inputs: "List[InputVar]", out_angle: float) -> "List[float]":
    """Return outward angles for children of a sub-model node.

    Fans the children in a cone centred on *out_angle*, with arc width
    proportional to leaf count (at least 1 per child), capped at
    ``_ARC_CAP_SUB_DEG``.  Each child inherits its own angle for further
    recursive sub-tree growth.
    """
    if not inputs:
        return []
    raw_counts = [_leaf_count(iv) for iv in inputs]
    total_leaves = sum(raw_counts)
    smoothed = [max(c, 1) for c in raw_counts]
    smoothed_total = sum(smoothed)

    arc_deg = min(_ARC_CAP_SUB_DEG, total_leaves * _ARC_PER_LEAF_SUB)
    arc_rad = math.radians(arc_deg)
    start = out_angle + arc_rad / 2
    angles: List[float] = []
    cumulative = 0.0
    for s in smoothed:
        frac = s / smoothed_total
        angles.append(start - (cumulative + frac / 2) * arc_rad)
        cumulative += frac
    return angles


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

    def abs_math_node(self, nid: str, style: str, math: str,
                      ref: str, dx: float, dy: float,
                      extra: str = "", ind: int = 2) -> None:
        """Emit a node at an absolute offset (dx, dy) cm from *ref*."""
        e = f", {extra}" if extra else ""
        self._lines.append(
            "\t" * ind
            + rf"\node [{style}{e}] at ($({ref})+({dx:.2f}cm,{dy:.2f}cm)$)"
            + rf" ({nid}) {{${math}$}};"
        )

    def text_node(self, nid: str, style: str, text: str,
                  pos: str = "", extra: str = "", ind: int = 2) -> None:
        p = f", {pos}" if pos else ""
        e = f", {extra}" if extra else ""
        self._lines.append(
            "\t" * ind + rf"\node [{style}{e}{p}] ({nid}) {{{text}}};"
        )

    def abs_text_node(self, nid: str, style: str, text: str,
                      ref: str, dx: float, dy: float,
                      extra: str = "", ind: int = 2) -> None:
        """Emit a text node at an absolute offset (dx, dy) cm from *ref*."""
        e = f", {extra}" if extra else ""
        self._lines.append(
            "\t" * ind
            + rf"\node [{style}{e}] at ($({ref})+({dx:.2f}cm,{dy:.2f}cm)$)"
            + rf" ({nid}) {{{text}}};"
        )

    def edge(self, src: str, dst: str, style: str, ind: int = 2) -> None:
        self._lines.append(
            "\t" * ind + rf"\draw [{style}] ({src}) -- ({dst});"
        )

    def get(self) -> str:
        return "\n".join(self._lines)


# ── Layout simulation & automatic overlap resolution ─────────────────────────

from collections import namedtuple as _nt

_NodeRecord = _nt("_NodeRecord", ["x", "y", "ntype", "ivar"])
# ntype values: 'deriv', 'model', 'leaf', 'effect'

# Approximate bounding-box half-sizes per node type (cm).
# Based on typical rendered sizes in \small/\footnotesize LaTeX fonts.
_BBOX_HALF: Dict[str, Tuple[float, float]] = {
    "deriv":  (0.95, 0.50),   # partial derivative fraction
    "model":  (2.00, 0.55),   # model equation (wide but not as wide as text)
    "leaf":   (0.75, 0.45),   # u(x) box (text width 1.1cm)
    "effect": (1.05, 0.75),   # multi-line italic text (text width 1.8cm)
}


def _aabb(rec: "_NodeRecord") -> Tuple[float, float, float, float]:
    hw, hh = _BBOX_HALF[rec.ntype]
    return rec.x - hw, rec.x + hw, rec.y - hh, rec.y + hh


def _aabb_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float]]:
    """Return (overlap_x, overlap_y) if AABBs overlap, else None."""
    ox = min(a[1], b[1]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[2], b[2])
    if ox > 0 and oy > 0:
        return ox, oy
    return None


def _simulate_branch(
    parent_model: "MeasurementModel",
    ivar: "InputVar",
    x_deriv: float,
    y_deriv: float,
    out_angle: float,
    depth: int = 0,
    cum_offset: Tuple[float, float] = (0.0, 0.0),
    root_ivar: "Optional[InputVar]" = None,
) -> "List[_NodeRecord]":
    """Return _NodeRecords for all nodes in this branch (mirrors _emit_branch).

    *root_ivar* is the top-level InputVar that owns this branch; when None it
    defaults to *ivar* itself.  All descendant records carry the same
    *root_ivar* so that the auto-layout can push entire branch subtrees as a
    unit rather than individual leaves.
    """
    if root_ivar is None:
        root_ivar = ivar
    scale = max(_DEPTH_SCALE ** depth, 0.5)
    cos_o = math.cos(out_angle)
    sin_o = math.sin(out_angle)
    eff_ox = cum_offset[0] + ivar.branch_offset[0]
    eff_oy = cum_offset[1] + ivar.branch_offset[1]

    recs: List["_NodeRecord"] = [
        _NodeRecord(x_deriv + eff_ox, y_deriv + eff_oy, "deriv", root_ivar)
    ]

    if ivar.submodel is not None:
        v_m0 = max(_V_M0 * scale, _V_M0_MIN)
        x_model_nat = x_deriv + v_m0 * cos_o
        y_model_nat = y_deriv + v_m0 * sin_o
        recs.append(_NodeRecord(x_model_nat + eff_ox, y_model_nat + eff_oy, "model", root_ivar))
        if not ivar.separate_figure:
            v_d1 = max(_V_D1 * scale, _V_D1_MIN)
            recs.extend(_simulate_inputs(
                ivar.submodel, x_model_nat, y_model_nat,
                v_d1, out_angle, depth + 1, (eff_ox, eff_oy),
                root_ivar=root_ivar,
            ))
    else:
        v_leaf = max(_V_LEAF * scale, _V_LEAF_MIN)
        x_leaf_nat = x_deriv + v_leaf * cos_o
        y_leaf_nat = y_deriv + v_leaf * sin_o
        recs.append(_NodeRecord(x_leaf_nat + eff_ox, y_leaf_nat + eff_oy, "leaf", root_ivar))
        if ivar.effects:
            recs.append(_NodeRecord(
                x_leaf_nat + _V_EFF * scale * cos_o + eff_ox,
                y_leaf_nat + _V_EFF * scale * sin_o + eff_oy,
                "effect", root_ivar,
            ))
    return recs


def _simulate_inputs(
    model: "MeasurementModel",
    parent_dx_nat: float,
    parent_dy_nat: float,
    v_to_child: float,
    out_angle: float,
    depth: int = 0,
    cum_offset: Tuple[float, float] = (0.0, 0.0),
    root_ivar: "Optional[InputVar]" = None,
) -> "List[_NodeRecord]":
    """Return _NodeRecords for all children of *model* (mirrors _emit_inputs).

    Children are placed in a radial sub-fan centred on *out_angle* so that
    each child grows in its own direction, preventing all branches from
    piling up in a single perpendicular direction.
    """
    if not model.inputs:
        return []

    recs: List["_NodeRecord"] = []
    for ivar, child_angle in zip(model.inputs, _sub_angles(model.inputs, out_angle)):
        x_d_nat = parent_dx_nat + v_to_child * math.cos(child_angle)
        y_d_nat = parent_dy_nat + v_to_child * math.sin(child_angle)
        recs.extend(_simulate_branch(model, ivar, x_d_nat, y_d_nat,
                                     child_angle, depth, cum_offset,
                                     root_ivar=root_ivar))
    return recs


def _auto_layout(model: "MeasurementModel", max_iterations: int = 40) -> int:
    """Iteratively adjust *branch_offset* on each :class:`InputVar` to eliminate
    bounding-box overlaps between nodes from different branches.

    Returns the number of iterations performed.  Modifies *model* in-place.
    """
    GAP  = 0.25   # minimum clearance gap added on top of each resolved overlap (cm)
    DAMP = 0.45   # damping factor — prevents oscillation

    angles = _root_angles(model.inputs)

    for iteration in range(max_iterations):
        # ── simulate current layout ───────────────────────────────────────────
        all_recs: List["_NodeRecord"] = []
        for ivar, angle in zip(model.inputs, angles):
            x_d = _V_D0 * math.cos(angle)
            y_d = _V_D0 * math.sin(angle)
            all_recs.extend(_simulate_branch(model, ivar, x_d, y_d, angle))

        # ── detect pairwise overlaps ──────────────────────────────────────────
        # Key: id(ivar) → [ivar_ref, push_x, push_y]
        push: Dict[int, List] = {}
        moved = False
        n = len(all_recs)
        for i in range(n):
            for j in range(i + 1, n):
                ri, rj = all_recs[i], all_recs[j]
                if ri.ivar is rj.ivar:
                    continue  # same branch — fine
                ov = _aabb_overlap(_aabb(ri), _aabb(rj))
                if ov is None:
                    continue

                ov_x, ov_y = ov
                dx = ri.x - rj.x
                dy = ri.y - rj.y
                dist = math.hypot(dx, dy) or 1e-6

                # Push proportional to overlap, damped, in the direction i→j
                px = DAMP * (ov_x + GAP) * dx / dist
                py = DAMP * (ov_y + GAP) * dy / dist

                push.setdefault(id(ri.ivar), [ri.ivar, 0.0, 0.0])
                push.setdefault(id(rj.ivar), [rj.ivar, 0.0, 0.0])
                push[id(ri.ivar)][1] += px / 2
                push[id(ri.ivar)][2] += py / 2
                push[id(rj.ivar)][1] -= px / 2
                push[id(rj.ivar)][2] -= py / 2
                moved = True

        if not moved:
            break

        for ivar_ref, px, py in push.values():
            ivar_ref.branch_offset = (
                ivar_ref.branch_offset[0] + px,
                ivar_ref.branch_offset[1] + py,
            )

    return iteration + 1


class _Emitter:
    """Walks the MeasurementModel tree and emits TikZ source.

    Root-level inputs are placed at angles computed by :func:`_root_angles`.
    Each sub-tree grows outward in its branch's angle; children are spread
    in the perpendicular direction (clockwise 90° from outward so the fan
    matches the standard left-to-right reading order when facing outward).
    """

    def __init__(self, tikz: _TikZ) -> None:
        self.t = tikz
        self._counters: Dict[str, int] = {}

    def _uid(self, base: str) -> str:
        """Return a unique TikZ node ID derived from *base*."""
        n = self._counters.get(base, 0)
        self._counters[base] = n + 1
        return base if n == 0 else f"{base}{n + 1}"

    # ── root ─────────────────────────────────────────────────────────────────

    def emit_root(self, model: MeasurementModel, root_id: str) -> None:
        """Emit the root block then radiate all branches at their angles."""
        self.t.comment(f"ROOT: {model.latex_name}")
        self.t.math_node(root_id, "root_block",
                         rf"{model.latex_name} = {model.latex_expr}")
        self.t.blank()
        for ivar, angle in zip(model.inputs, _root_angles(model.inputs)):
            x_d = _V_D0 * math.cos(angle)
            y_d = _V_D0 * math.sin(angle)
            self._emit_branch(model, root_id, ivar,
                              x_d, y_d, root_id, angle, depth=0,
                              cum_offset=(0.0, 0.0))

    # ── single branch ────────────────────────────────────────────────────────

    def _emit_branch(self, parent_model: MeasurementModel,
                     parent_id: str, ivar: InputVar,
                     x_deriv: float, y_deriv: float,
                     root_id: str, out_angle: float,
                     depth: int = 0,
                     cum_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        """Emit  parent → deriv_node → [sub-model children | leaf → effects].

        *out_angle* is the angle (radians) pointing away from the root along
        this branch.  Sub-model children fan out in a radial cone centred on
        *out_angle* (see :func:`_sub_angles`), each inheriting their own angle.
        *depth* drives adaptive spacing: distances shrink by _DEPTH_SCALE per level.
        *cum_offset* accumulates branch_offsets from ancestor branches; this
        branch adds its own *ivar.branch_offset* on top.
        """
        scale = max(_DEPTH_SCALE ** depth, 0.5)  # floor at 50% to avoid extreme compression
        eff_ox = cum_offset[0] + ivar.branch_offset[0]
        eff_oy = cum_offset[1] + ivar.branch_offset[1]
        color = ivar.color
        cos_o = math.cos(out_angle)
        sin_o = math.sin(out_angle)
        deriv_lat = _deriv_label(parent_model.latex_name, ivar.latex_name)
        d_id = self._uid(f"D{_tikz_id(ivar.latex_name)}")

        self.t.comment(f"∂{parent_model.latex_name}/∂{ivar.latex_name}")
        self.t.abs_math_node(d_id, "deriv_node", deriv_lat,
                             ref=root_id, dx=x_deriv + eff_ox, dy=y_deriv + eff_oy)
        self.t.edge(parent_id, d_id, f"connection, {color}")

        if ivar.submodel is not None:
            m_id = self._uid(f"M{_tikz_id(ivar.latex_name)}")
            v_m0 = max(_V_M0 * scale, _V_M0_MIN)
            x_model_nat = x_deriv + v_m0 * cos_o
            y_model_nat = y_deriv + v_m0 * sin_o
            self.t.blank()
            self.t.comment(f"sub-model: {ivar.submodel.latex_name}")
            if ivar.separate_figure:
                # Show model equation + cross-ref only — no u(x) leaf needed.
                ref_note = (rf"${ivar.submodel.latex_name} = {ivar.submodel.latex_expr}$"
                            rf"\\ \footnotesize(see Fig.~\ref{{fig:{ivar.separate_label}}})")
                self.t.abs_text_node(
                    m_id, "model_block",
                    ref_note,
                    ref=root_id, dx=x_model_nat + eff_ox, dy=y_model_nat + eff_oy,
                )
                self.t.edge(d_id, m_id, f"connection, {color}")
            else:
                self.t.abs_math_node(
                    m_id, "model_block",
                    rf"{ivar.submodel.latex_name} = {ivar.submodel.latex_expr}",
                    ref=root_id, dx=x_model_nat + eff_ox, dy=y_model_nat + eff_oy,
                )
                self.t.edge(d_id, m_id, f"connection, {color}")
                v_d1 = max(_V_D1 * scale, _V_D1_MIN)
                self._emit_inputs(ivar.submodel, m_id, x_model_nat, y_model_nat,
                                  root_id, v_d1, out_angle,
                                  depth=depth + 1, cum_offset=(eff_ox, eff_oy))
        else:
            leaf_id = self._uid(f"U{_tikz_id(ivar.latex_name)}")
            v_leaf = max(_V_LEAF * scale, _V_LEAF_MIN)
            x_leaf_nat = x_deriv + v_leaf * cos_o
            y_leaf_nat = y_deriv + v_leaf * sin_o
            self.t.abs_math_node(
                leaf_id, "leaf_node", rf"u({ivar.latex_name})",
                ref=root_id, dx=x_leaf_nat + eff_ox, dy=y_leaf_nat + eff_oy,
                extra=f"draw={color}, text={color}",
            )
            self.t.edge(d_id, leaf_id, f"connection, {color}")
            self._emit_effects(ivar, leaf_id,
                               root_id=root_id,
                               dx=x_leaf_nat + eff_ox, dy=y_leaf_nat + eff_oy,
                               out_angle=out_angle,
                               scale=scale)
        self.t.blank()

    # ── sub-model children ───────────────────────────────────────────────────

    def _emit_inputs(self, model: MeasurementModel,
                     parent_id: str,
                     parent_dx: float, parent_dy: float,
                     root_id: str,
                     v_to_child: float,
                     out_angle: float,
                     depth: int = 0,
                     cum_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        """Fan all inputs of *model* outward in a radial sub-fan from *parent_id*.

        Each child gets its own angle from :func:`_sub_angles`, centred on
        *out_angle*, so sub-trees grow in distinct directions instead of all
        piling into one perpendicular direction.
        *depth* is passed to _emit_branch for adaptive spacing.
        *cum_offset* is the accumulated branch_offset from all ancestor branches.
        """
        inputs = model.inputs
        if not inputs:
            return

        for ivar, child_angle in zip(inputs, _sub_angles(inputs, out_angle)):
            x_d = parent_dx + v_to_child * math.cos(child_angle)
            y_d = parent_dy + v_to_child * math.sin(child_angle)
            self._emit_branch(model, parent_id, ivar,
                              x_d, y_d, root_id, child_angle, depth=depth,
                              cum_offset=cum_offset)

    # ── effect nodes ─────────────────────────────────────────────────────────

    def _emit_effects(self, ivar: InputVar, leaf_id: str,
                      root_id: str, dx: float, dy: float,
                      out_angle: float, scale: float = 1.0) -> None:
        """Emit uncertainty-source nodes further outward from the leaf."""
        if not ivar.effects:
            return
        eff_id = self._uid(f"EFF{_tikz_id(ivar.latex_name)}")
        eff_text = r" \\ ".join(ivar.effects)
        x_eff = dx + _V_EFF * scale * math.cos(out_angle)
        y_eff = dy + _V_EFF * scale * math.sin(out_angle)
        self.t.abs_text_node(eff_id, "effect_node", eff_text,
                             ref=root_id, dx=x_eff, dy=y_eff,
                             extra=ivar.color)
        self.t.edge(eff_id, leaf_id, f"connection, {ivar.color}, dashed")


def collect_separate_figures(
    model: MeasurementModel,
) -> List[tuple]:  # list of (InputVar, MeasurementModel) pairs
    """Recursively collect all sub-models marked for separate figures.

    Returns a flat list of ``(ivar, submodel)`` pairs in depth-first order,
    where *ivar.separate_figure* is True.  The caller can then call
    :func:`build_tikz` for each sub-model using ``ivar.separate_label``.
    """
    result = []
    for ivar in model.inputs:
        if ivar.submodel is not None:
            if ivar.separate_figure:
                result.append((ivar, ivar.submodel))
                # Also recurse into the separate sub-model itself
                result.extend(collect_separate_figures(ivar.submodel))
            else:
                result.extend(collect_separate_figures(ivar.submodel))
    return result


def _walk_inputs(model: "MeasurementModel") -> "List[InputVar]":
    """Return all InputVar objects in the tree (depth-first)."""
    result: List[InputVar] = []
    for ivar in model.inputs:
        result.append(ivar)
        if ivar.submodel is not None and not ivar.separate_figure:
            result.extend(_walk_inputs(ivar.submodel))
    return result


def build_tikz(root: MeasurementModel, label: str = "",
               caption: str = "",
               auto_layout: bool = True) -> str:
    """Return a complete LaTeX figure environment with the TikZ UTD.

    Parameters
    ----------
    root:
        The root measurement model.
    label:
        LaTeX ``\\label`` key used inside the figure (without the
        ``fig:`` prefix).  Defaults to ``utd_<ROOTSYM>``.
    caption:
        Caption text.  Defaults to
        ``Uncertainty Tree Diagram for $<root.latex_name>$.``
    auto_layout:
        When True (default), run :func:`_auto_layout` before generating
        TikZ to automatically resolve bounding-box overlaps between branches.
        Set to False in tests or when branch_offsets are already hand-tuned.
    """
    if auto_layout:
        iters = _auto_layout(root)
        adjusted = [iv for iv in _walk_inputs(root) if iv.branch_offset != (0.0, 0.0)]
        if adjusted:
            names = ", ".join(iv.latex_name for iv in adjusted)
            print(f"  ↻  Auto-layout: {iters} iteration(s); adjusted: {names}")
        else:
            print(f"  ↻  Auto-layout: {iters} iteration(s); no overlaps detected")
    if not label:
        label = f"utd_{_tikz_id(root.latex_name).lower()}"
    if not caption:
        caption = rf"Uncertainty Tree Diagram for ${root.latex_name}$."
    t = _TikZ()
    # Requires in the including document:
    #   \usepackage{tikz}
    #   \usetikzlibrary{positioning,calc}
    #   \usepackage{amsmath,amssymb,xcolor,float,adjustbox}
    t.raw(r"\begin{figure}[p]")
    t.raw(r"  \centering")
    t.raw(r"  % Scale to fit page: preserves aspect ratio within textwidth × 0.88 textheight")
    t.raw(r"  \begin{adjustbox}{max width=\textwidth, max totalheight=.88\textheight, keepaspectratio}")
    t.raw(r"  \begin{tikzpicture}[")
    t.raw(r"    connection/.style={draw, thick},")
    t.raw(r"    root_block/.style={draw, rectangle, inner sep=6pt,"
          r" font=\normalsize\bfseries, align=center},")
    t.raw(r"    model_block/.style={draw, rectangle, inner sep=4pt,"
          r" font=\small\bfseries, align=center},")
    t.raw(r"    deriv_node/.style={draw, rectangle, rounded corners=3pt,"
          r" inner sep=3pt, font=\small, align=center},")
    t.raw(r"    leaf_node/.style={draw, rectangle, inner sep=3pt,"
          r" font=\footnotesize, align=center, text width=1.1cm},")
    t.raw(r"    effect_node/.style={draw, dashed, font=\scriptsize\itshape,"
          r" align=center, text width=1.8cm, inner sep=2pt}")
    t.raw(r"    ]")

    root_id = _tikz_id(root.latex_name) + "ROOT"
    emitter = _Emitter(t)
    emitter.emit_root(root, root_id)

    t.raw(r"  \end{tikzpicture}")
    t.raw(r"  \end{adjustbox}")
    t.raw(rf"  \caption{{{caption}}}")
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

def _strip_latex_command(text: str, cmd: str) -> str:
    """Remove all occurrences of \\cmd{...} (with balanced braces) from *text*."""
    result = []
    i = 0
    pattern = re.compile(r'\s*\\' + re.escape(cmd) + r'\s*\{')
    while i < len(text):
        m = pattern.search(text, i)
        if not m:
            result.append(text[i:])
            break
        result.append(text[i:m.start()])
        # Walk forward to find the matching closing brace
        j = m.end()  # position after opening '{'
        depth = 1
        while j < len(text) and depth:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        # Skip optional trailing newline
        if j < len(text) and text[j] == '\n':
            j += 1
        i = j
    return ''.join(result)

# Minimal LaTeX wrapper that compiles a bare \begin{figure}…\end{figure} snippet.
_LATEX_WRAPPER = r"""
\documentclass[border=6pt]{{standalone}}
\usepackage{{tikz}}
\usetikzlibrary{{positioning, calc}}
\usepackage{{amsmath, amssymb}}
\usepackage{{xcolor}}
\usepackage{{float}}
\usepackage{{adjustbox}}
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
    # Use brace-balanced strip for \caption{…} so nested {} (e.g. $\varpi_{\rm g}$)
    # are handled correctly.
    body_inner = _strip_latex_command(body_inner, "caption")
    body_inner = re.sub(r"\s*\\label\{[^}]*\}\n?", "", body_inner)
    body_inner = re.sub(r"\\centering\n?", "", body_inner)
    # Strip adjustbox wrapper — standalone auto-sizes; adjustbox is a no-op there
    body_inner = re.sub(r"\\begin\{adjustbox\}[^\n]*\n?", "", body_inner)
    body_inner = re.sub(r"\\end\{adjustbox\}\n?", "", body_inner)
    # Strip comment lines added by build_tikz
    body_inner = re.sub(r"\s*%[^\n]*\n", "\n", body_inner)

    wrapper = _LATEX_WRAPPER.format(body=body_inner.strip())

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "fig.tex")
        pdf = os.path.join(td, "fig.pdf")
        Path(src).write_text(wrapper)

        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", td, src],
            capture_output=True, text=True,
        )
        if not os.path.exists(pdf):
            print("  ✗  pdflatex failed (no PDF produced):", file=sys.stderr)
            for line in result.stdout.splitlines()[-20:]:
                print("     " + line, file=sys.stderr)
            return False
        if result.returncode != 0:
            # PDF was produced but with warnings/errors — show them and continue
            errors = [l for l in result.stdout.splitlines() if l.startswith("!")]
            if errors:
                print("  ⚠  pdflatex warnings:", file=sys.stderr)
                for line in errors[:5]:
                    print("     " + line, file=sys.stderr)

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
        caption = ""  # use build_tikz default
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
        default_caption = rf"Uncertainty Tree Diagram for ${lat_name}$."
        caption = _ask("  Figure caption", default_caption)
        print()
        print("Step 2: Measurement model")
        symtable: Dict[str, sp.Symbol] = {}
        color_pool = list(COLORS)
        model = collect_model(lat_name, symtable, color_pool)

    tikz_code = build_tikz(model, label=label, caption=caption)

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
