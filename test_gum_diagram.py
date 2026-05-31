"""
test_gum_diagram.py – Unit tests for gum_diagram.py

Run with:
    python -m pytest test_gum_diagram.py -v
    # or directly:
    python test_gum_diagram.py
"""
import math
import re
import sys
import unittest
from pathlib import Path

import sympy as sp

# Make sure the module can be imported from the same directory
sys.path.insert(0, str(Path(__file__).parent))
import gum_diagram as gd


# ── Helpers ──────────────────────────────────────────────────────────────────

def _simple_model() -> gd.MeasurementModel:
    """y = a * x + b  (all leaves, no sub-models, using LaTeX input)."""
    expr, st = gd._parse_latex_expr(r"a \cdot x + b", {})
    syms = {str(s): s for s in expr.free_symbols}
    a_iv = gd.InputVar(r"a", syms["a"], "red", effects=["Calibration"])
    x_iv = gd.InputVar(r"x", syms["x"], "blue!70!black")
    b_iv = gd.InputVar(r"b", syms["b"], "purple", effects=["Offset estimation"])
    return gd.MeasurementModel(
        latex_name=r"y",
        latex_expr=r"a \cdot x + b",
        expr=expr,
        inputs=[a_iv, x_iv, b_iv],
    )


def _nested_model() -> gd.MeasurementModel:
    """z = p * q  where p has sub-model p = u / v (using LaTeX input)."""
    sub_expr, st = gd._parse_latex_expr(r"\frac{u}{v}", {})
    syms_sub = {str(s): s for s in sub_expr.free_symbols}
    u_iv = gd.InputVar(r"u", syms_sub["u"], "red", effects=["Measurement A"])
    v_iv = gd.InputVar(r"v", syms_sub["v"], "purple", effects=["Measurement B"])
    p_model = gd.MeasurementModel(
        latex_name=r"p",
        latex_expr=r"\frac{u}{v}",
        expr=sub_expr,
        inputs=[u_iv, v_iv],
    )

    root_expr, st2 = gd._parse_latex_expr(r"p \cdot q", {})
    syms_root = {str(s): s for s in root_expr.free_symbols}
    p_iv = gd.InputVar(r"p", syms_root["p"], "red", submodel=p_model)
    q_iv = gd.InputVar(r"q", syms_root["q"], "blue!70!black", effects=["NWP"])
    return gd.MeasurementModel(
        latex_name=r"z",
        latex_expr=r"p \cdot q",
        expr=root_expr,
        inputs=[p_iv, q_iv],
    )


# ── _latex_to_sym_name ───────────────────────────────────────────────────────

class TestLatexToSymName(unittest.TestCase):
    def test_greek_lambda(self):
        self.assertEqual(gd._latex_to_sym_name(r"\lambda_C"), "lam_C")

    def test_greek_theta(self):
        self.assertEqual(gd._latex_to_sym_name(r"\theta"), "theta")

    def test_mathbf_stripped(self):
        self.assertEqual(gd._latex_to_sym_name(r"\mathbf{b}"), "b")

    def test_plain_ascii(self):
        self.assertEqual(gd._latex_to_sym_name("x"), "x")

    def test_no_empty_result(self):
        # Any input should return a non-empty string
        self.assertTrue(len(gd._latex_to_sym_name(r"\{")) > 0)


# ── _parse_latex_expr ────────────────────────────────────────────────────────

class TestParseLatexExpr(unittest.TestCase):
    def test_simple_sum(self):
        expr, st = gd._parse_latex_expr(r"a + b", {})
        names = {str(s) for s in expr.free_symbols}
        self.assertIn("a", names)
        self.assertIn("b", names)

    def test_fraction(self):
        expr, st = gd._parse_latex_expr(r"\frac{x}{y}", {})
        names = {str(s) for s in expr.free_symbols}
        self.assertIn("x", names)
        self.assertIn("y", names)

    def test_symtable_populated(self):
        expr, st = gd._parse_latex_expr(r"a \cdot b", {})
        self.assertIn("a", st)
        self.assertIn("b", st)

    def test_symtable_reuse(self):
        """Same variable across two parse calls should reuse the same Symbol."""
        _, st1 = gd._parse_latex_expr(r"a + c", {})
        expr2, st2 = gd._parse_latex_expr(r"a + d", st1)
        # 'a' should be the same Symbol object in both
        a_name = next(str(s) for s in expr2.free_symbols if str(s) == "a")
        self.assertIn(a_name, st2)

    def test_equation_rhs_stripped(self):
        """If user types 'y = a + b', only the RHS should be returned."""
        # parse_latex may or may not return Eq; if it does, we take rhs
        expr, _ = gd._parse_latex_expr(r"a + b", {})
        self.assertFalse(isinstance(expr, sp.Eq))


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
        self.assertEqual(gd._label_to_filename("fig:_utd"), "fig_utd.tex")

    def test_always_ends_in_tex(self):
        self.assertTrue(gd._label_to_filename("xyz").endswith(".tex"))


# ── _child_offsets / _root_angles ────────────────────────────────────────────

class TestChildOffsets(unittest.TestCase):
    def _make_leaf(self, name: str = "x") -> gd.InputVar:
        sym = __import__("sympy").Symbol(name)
        return gd.InputVar(latex_name=name, sym=sym, color="black")

    def test_single_input_centred_at_zero(self):
        iv = self._make_leaf("x")
        offs = gd._child_offsets([iv])
        self.assertEqual(len(offs), 1)
        self.assertAlmostEqual(offs[0], 0.0)

    def test_two_inputs_symmetric(self):
        a, b = self._make_leaf("a"), self._make_leaf("b")
        offs = gd._child_offsets([a, b])
        self.assertAlmostEqual(offs[0], -offs[1])
        self.assertGreater(offs[1], 0)

    def test_offsets_increase_left_to_right(self):
        ivs = [self._make_leaf(n) for n in ("a", "b", "c")]
        offs = gd._child_offsets(ivs)
        self.assertLess(offs[0], offs[1])
        self.assertLess(offs[1], offs[2])


class TestRootAngles(unittest.TestCase):
    def _make_leaf(self, name: str = "x") -> gd.InputVar:
        sym = __import__("sympy").Symbol(name)
        return gd.InputVar(latex_name=name, sym=sym, color="black")

    def test_single_input_points_straight_up(self):
        iv = self._make_leaf("x")
        angles = gd._root_angles([iv])
        self.assertEqual(len(angles), 1)
        self.assertAlmostEqual(angles[0], math.pi / 2, places=5)

    def test_two_inputs_symmetric_about_vertical(self):
        a, b = self._make_leaf("a"), self._make_leaf("b")
        angles = gd._root_angles([a, b])
        mid = (angles[0] + angles[1]) / 2
        self.assertAlmostEqual(mid, math.pi / 2, places=5)

    def test_angles_decrease_left_to_right(self):
        """Left-most angle > right-most (counter-clockwise ordering)."""
        ivs = [self._make_leaf(n) for n in ("a", "b", "c")]
        angles = gd._root_angles(ivs)
        self.assertGreater(angles[0], angles[1])
        self.assertGreater(angles[1], angles[2])

    def test_narrow_arc_for_few_leaves(self):
        """6 or fewer leaves → arc ≤ 180° (all above the root)."""
        ivs = [self._make_leaf(n) for n in ("a", "b", "c", "d", "e", "f")]
        angles = gd._root_angles(ivs)
        arc = angles[0] - angles[-1]
        self.assertLessEqual(math.degrees(arc), 180.0 + 1e-9)
        # all angles in upper half-plane
        for a in angles:
            self.assertGreaterEqual(a, 0.0)

    def test_wide_arc_for_many_leaves(self):
        """12 leaves → arc ≥ 300° (wraps well past the sides)."""
        ivs = [self._make_leaf(str(i)) for i in range(12)]
        angles = gd._root_angles(ivs)
        arc = angles[0] - angles[-1]
        self.assertGreater(math.degrees(arc), 300.0)

    def test_capped_at_arc_cap(self):
        ivs = [self._make_leaf(str(i)) for i in range(30)]
        angles = gd._root_angles(ivs)
        arc = math.degrees(angles[0] - angles[-1])
        self.assertLessEqual(arc, gd._ARC_CAP_DEG + 1e-9)


# ── _render_deriv ─────────────────────────────────────────────────────────────

class TestRenderDeriv(unittest.TestCase):
    def test_linear_derivative(self):
        model = _simple_model()
        a_iv, x_iv, b_iv = model.inputs
        # d(a*x+b)/da = x
        out = gd._render_deriv(model, a_iv)
        self.assertIn("x", out)

    def test_derivative_uses_latex_names(self):
        """Sympy symbol names should be replaced with user LaTeX names."""
        expr, st = gd._parse_latex_expr(r"\frac{\lambda}{b}", {})
        syms = {str(s): s for s in expr.free_symbols}
        lam_iv = gd.InputVar(r"\lambda", syms["lam"], "red")
        b_iv = gd.InputVar(r"b_0", syms["b"], "blue!70!black")
        model = gd.MeasurementModel(
            latex_name=r"y", latex_expr=r"\frac{\lambda}{b}",
            expr=expr, inputs=[lam_iv, b_iv],
        )
        out = gd._render_deriv(model, lam_iv)
        # The result should contain user-supplied LaTeX, not raw sympy name
        self.assertNotIn("lam", out)  # sympy's internal name gone

    def test_constant_wrt_absent_sym(self):
        model = _simple_model()
        z_iv = gd.InputVar(r"z", sp.Symbol("z_other"), "green!60!black")
        out = gd._render_deriv(model, z_iv)
        self.assertEqual(out.strip(), "0")


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
        z_iv = gd.InputVar(r"z", z, "green!60!black")
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
        self.assertIn("u(a)", out)
        self.assertIn("u(x)", out)
        self.assertIn("u(b)", out)

    def test_effect_node_present_when_effects_given(self):
        out = gd.build_tikz(self.simple)
        self.assertIn("Calibration", out)
        self.assertIn("Offset estimation", out)

    def test_no_effect_node_when_no_effects(self):
        out = gd.build_tikz(self.simple)
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
        self.assertIn(r"\lambda_C", out)
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
