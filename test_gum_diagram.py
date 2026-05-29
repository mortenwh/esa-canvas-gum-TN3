"""
test_gum_diagram.py – Unit tests for gum_diagram.py

Run with:
    python -m pytest test_gum_diagram.py -v
    # or directly:
    python test_gum_diagram.py
"""
import re
import sys
import textwrap
import unittest
from pathlib import Path

import sympy as sp

# Make sure the module can be imported from the same directory
sys.path.insert(0, str(Path(__file__).parent))
import gum_diagram as gd


# ── Helpers ──────────────────────────────────────────────────────────────────

def _simple_model() -> gd.MeasurementModel:
    """y = a * x + b  (all leaves, no sub-models)."""
    a, x, b = sp.symbols("a x b")
    a_iv = gd.InputVar(a, r"a", "red", effects=["Calibration"])
    x_iv = gd.InputVar(x, r"x", "blue!70!black")
    b_iv = gd.InputVar(b, r"b", "purple", effects=["Offset estimation"])
    return gd.MeasurementModel(
        sym=sp.Symbol("y"),
        latex_name=r"y",
        expr=a * x + b,
        inputs=[a_iv, x_iv, b_iv],
    )


def _nested_model() -> gd.MeasurementModel:
    """z = p * q  where p has sub-model p = u / v."""
    u, v, q = sp.symbols("u v q")
    u_iv = gd.InputVar(u, r"u", "red", effects=["Measurement A"])
    v_iv = gd.InputVar(v, r"v", "purple", effects=["Measurement B"])
    p_expr = u / v
    p_model = gd.MeasurementModel(
        sym=sp.Symbol("p"), latex_name=r"p",
        expr=p_expr, inputs=[u_iv, v_iv],
    )
    p_iv = gd.InputVar(sp.Symbol("p"), r"p", "red", submodel=p_model)
    q_iv = gd.InputVar(q, r"q", "blue!70!black", effects=["NWP"])
    p_sym, q_sym = sp.symbols("p q")
    return gd.MeasurementModel(
        sym=sp.Symbol("z"),
        latex_name=r"z",
        expr=p_sym * q_sym,
        inputs=[p_iv, q_iv],
    )


# ── _tikz_id ─────────────────────────────────────────────────────────────────

class TestTikzId(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(gd._tikz_id("H_s"), "HS")

    def test_special_chars_stripped(self):
        self.assertEqual(gd._tikz_id("lam_C20"), "LAMC20")

    def test_backslash_stripped(self):
        self.assertEqual(gd._tikz_id(r"\lambda"), "LAMBDA")

    def test_digits_kept(self):
        self.assertEqual(gd._tikz_id("x1"), "X1")


# ── _label_to_filename ───────────────────────────────────────────────────────

class TestLabelToFilename(unittest.TestCase):
    def test_colon_replaced(self):
        self.assertEqual(gd._label_to_filename("fig:swh_utd"), "fig_swh_utd.tex")

    def test_lowercase(self):
        self.assertEqual(gd._label_to_filename("SWH_UTD"), "swh_utd.tex")

    def test_spaces_replaced(self):
        self.assertEqual(gd._label_to_filename("my label"), "my_label.tex")

    def test_double_underscores_collapsed(self):
        # Colon → underscore, adjacent to another underscore → collapses
        self.assertEqual(gd._label_to_filename("fig:_utd"), "fig_utd.tex")

    def test_always_ends_in_tex(self):
        self.assertTrue(gd._label_to_filename("xyz").endswith(".tex"))


# ── _fan_pos ─────────────────────────────────────────────────────────────────

class TestFanPos(unittest.TestCase):
    def test_single_input_goes_straight_above(self):
        pos = gd._fan_pos(0, 1, "NODE")
        self.assertIn("above=", pos)
        self.assertNotIn("left", pos)
        self.assertNotIn("right", pos)

    def test_two_inputs_symmetric(self):
        left = gd._fan_pos(0, 2, "NODE")
        right = gd._fan_pos(1, 2, "NODE")
        self.assertIn("left", left)
        self.assertIn("right", right)
        # Extract the horizontal offset values and check they are equal
        l_cm = float(re.search(r"and ([\d.]+)cm", left).group(1))
        r_cm = float(re.search(r"and ([\d.]+)cm", right).group(1))
        self.assertAlmostEqual(l_cm, r_cm)

    def test_anchor_name_embedded(self):
        pos = gd._fan_pos(0, 3, "MYNODE")
        self.assertIn("MYNODE", pos)


# ── _expr_latex ───────────────────────────────────────────────────────────────

class TestExprLatex(unittest.TestCase):
    def test_simple_substitution(self):
        model = _simple_model()
        lat = gd._expr_latex(model)
        # sympy renders Symbol("a") as "a"; user provided "a" → no real change
        # but the expression should contain the user latex names
        self.assertIn("a", lat)
        self.assertIn("b", lat)
        self.assertIn("x", lat)

    def test_no_corruption_of_lambda(self):
        """A single-char symbol name must not corrupt \\lambda in the output."""
        lam, b = sp.symbols("lam b")
        lam_iv = gd.InputVar(lam, r"\lambda", "red")
        b_iv = gd.InputVar(b, r"b_{\rm cal}", "blue!70!black")
        model = gd.MeasurementModel(
            sym=sp.Symbol("y"), latex_name=r"y",
            expr=lam + b, inputs=[lam_iv, b_iv],
        )
        lat = gd._expr_latex(model)
        # \lambda must appear intact
        self.assertIn(r"\lambda", lat)
        # There should be no \lam that is not followed by b (i.e. \lambda intact)
        self.assertNotIn(r"\lamb_", lat)

    def test_long_name_before_short(self):
        """lam_C20 → \\lambda_C^{20} must be replaced before any single-char."""
        lam_C20, b = sp.symbols("lam_C20 b")
        l_iv = gd.InputVar(lam_C20, r"\lambda_C^{20}", "red")
        b_iv = gd.InputVar(b, r"\mathbf{b}", "blue!70!black")
        model = gd.MeasurementModel(
            sym=sp.Symbol("y"), latex_name=r"y",
            expr=lam_C20 + b, inputs=[l_iv, b_iv],
        )
        lat = gd._expr_latex(model)
        self.assertIn(r"\lambda_C^{20}", lat)
        self.assertIn(r"\mathbf{b}", lat)
        # The raw sympy form lam_{C20} should be fully replaced
        self.assertNotIn("lam_{C20}", lat)


# ── MeasurementModel.deriv_of ─────────────────────────────────────────────────

class TestDerivOf(unittest.TestCase):
    def test_linear_model(self):
        model = _simple_model()
        a_iv, x_iv, b_iv = model.inputs
        # y = a*x + b  →  dy/da = x
        self.assertEqual(sp.simplify(model.deriv_of(a_iv) - x_iv.sym), 0)
        # dy/db = 1
        self.assertEqual(sp.simplify(model.deriv_of(b_iv) - 1), 0)

    def test_constant_derivative_is_zero_for_absent_symbol(self):
        model = _simple_model()
        z = sp.Symbol("z_other")
        z_iv = gd.InputVar(z, r"z", "green!60!black")
        self.assertEqual(sp.diff(model.expr, z_iv.sym), 0)


# ── build_tikz structure ──────────────────────────────────────────────────────

class TestBuildTikz(unittest.TestCase):

    def setUp(self):
        self.simple = _simple_model()
        self.nested = _nested_model()

    # ── Boilerplate presence ────────────────────────────────────────────────
    def test_begins_with_figure(self):
        out = gd.build_tikz(self.simple)
        self.assertTrue(out.strip().startswith(r"\begin{figure}"))

    def test_ends_with_figure(self):
        out = gd.build_tikz(self.simple)
        self.assertTrue(out.strip().endswith(r"\end{figure}"))

    def test_contains_tikzpicture(self):
        out = gd.build_tikz(self.simple)
        self.assertIn(r"\begin{tikzpicture}", out)
        self.assertIn(r"\end{tikzpicture}", out)

    def test_contains_required_styles(self):
        out = gd.build_tikz(self.simple)
        for style in ("root_block", "model_block", "deriv_node",
                      "leaf_node", "effect_node"):
            with self.subTest(style=style):
                self.assertIn(style, out)

    # ── Label ──────────────────────────────────────────────────────────────
    def test_default_label(self):
        out = gd.build_tikz(self.simple)
        self.assertIn(r"\label{fig:utd_y}", out)

    def test_custom_label(self):
        out = gd.build_tikz(self.simple, label="my_fig")
        self.assertIn(r"\label{fig:my_fig}", out)

    def test_caption_contains_measurand(self):
        out = gd.build_tikz(self.simple)
        self.assertIn(r"$y$", out)

    # ── Root block ─────────────────────────────────────────────────────────
    def test_root_block_node_present(self):
        out = gd.build_tikz(self.simple)
        self.assertIn("root_block", out)
        self.assertRegex(out, r"\\node \[root_block\]")

    # ── Leaf inputs ────────────────────────────────────────────────────────
    def test_leaf_uncertainty_nodes(self):
        out = gd.build_tikz(self.simple)
        # u(a), u(x), u(b) should all appear
        self.assertIn("u(a)", out)
        self.assertIn("u(x)", out)
        self.assertIn("u(b)", out)

    def test_effect_node_present_when_effects_given(self):
        out = gd.build_tikz(self.simple)
        self.assertIn("Calibration", out)
        self.assertIn("Offset estimation", out)

    def test_no_effect_node_when_no_effects(self):
        out = gd.build_tikz(self.simple)
        # x_iv has no effects; no effect_node should appear adjacent to u(x)
        # We check that effect_node count matches the number of inputs with effects
        n_effect_nodes = out.count("effect_node,")
        # simple model: a has 1 effect, x has 0, b has 1 → 2 effect nodes
        self.assertEqual(n_effect_nodes, 2)

    def test_leaf_colors_applied(self):
        out = gd.build_tikz(self.simple)
        self.assertIn("draw=red", out)
        self.assertIn("draw=blue!70!black", out)
        self.assertIn("draw=purple", out)

    # ── Connections ─────────────────────────────────────────────────────────
    def test_connections_drawn(self):
        out = gd.build_tikz(self.simple)
        self.assertRegex(out, r"\\draw \[connection")

    def test_dashed_connections_for_effect_nodes(self):
        out = gd.build_tikz(self.simple)
        self.assertIn("dashed", out)

    # ── Partial derivative notation ─────────────────────────────────────────
    def test_partial_derivatives_present(self):
        out = gd.build_tikz(self.simple)
        self.assertIn(r"\frac{\partial y}{\partial a}", out)
        self.assertIn(r"\frac{\partial y}{\partial x}", out)
        self.assertIn(r"\frac{\partial y}{\partial b}", out)

    # ── Nested / sub-model ──────────────────────────────────────────────────
    def test_submodel_block_present(self):
        out = gd.build_tikz(self.nested)
        self.assertRegex(out, r"\\node \[model_block")

    def test_submodel_inputs_appear(self):
        out = gd.build_tikz(self.nested)
        self.assertIn("u(u)", out)
        self.assertIn("u(v)", out)

    def test_side_leaf_present(self):
        """q is a leaf input at root level → placed to the right."""
        out = gd.build_tikz(self.nested)
        self.assertIn("u(q)", out)

    def test_no_duplicate_node_ids(self):
        """Every TikZ node identifier must be unique."""
        out = gd.build_tikz(self.nested)
        ids = re.findall(r"\\node\s*\[[^\]]*\]\s*\(([^)]+)\)", out)
        self.assertEqual(len(ids), len(set(ids)), f"Duplicate TikZ IDs: {ids}")

    # ── Built-in example ────────────────────────────────────────────────────
    def test_builtin_example_runs(self):
        model = gd._builtin_example()
        out = gd.build_tikz(model, label="swh_utd")
        self.assertIn(r"\begin{figure}", out)
        self.assertIn(r"\label{fig:swh_utd}", out)

    def test_builtin_example_lambda_not_corrupted(self):
        model = gd._builtin_example()
        out = gd.build_tikz(model)
        # \lambda_C should appear intact in derivative notation
        self.assertIn(r"\lambda_C", out)
        # No garbled version like \lam\ followed by mathbf
        self.assertNotIn(r"\lam" + "\\", out)


# ── _label_to_filename round-trip ─────────────────────────────────────────────

class TestFilenameRoundTrip(unittest.TestCase):
    def test_example_label(self):
        self.assertEqual(gd._label_to_filename("swh_utd"), "swh_utd.tex")

    def test_fig_prefix(self):
        fn = gd._label_to_filename("fig:swh_utd")
        self.assertTrue(fn.endswith(".tex"))
        self.assertNotIn(":", fn)


# ── Integration: file written with correct label ──────────────────────────────

class TestFileOutput(unittest.TestCase):
    def test_output_file_named_after_label(self):
        import tempfile, os
        model = _simple_model()
        label = "test_y_utd"
        expected_name = gd._label_to_filename(label)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, expected_name)
            with open(path, "w") as fh:
                fh.write(gd.build_tikz(model, label=label) + "\n")
            self.assertTrue(os.path.exists(path))
            content = Path(path).read_text()
            self.assertIn(rf"\label{{fig:{label}}}", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
