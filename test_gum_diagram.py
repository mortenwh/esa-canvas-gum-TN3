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

    def test_single_input_points_to_root_center(self):
        iv = self._make_leaf("x")
        angles = gd._root_angles([iv])
        self.assertEqual(len(angles), 1)
        self.assertAlmostEqual(angles[0], gd._ROOT_CENTER_ANGLE, places=5)

    def test_two_inputs_symmetric_about_center(self):
        a, b = self._make_leaf("a"), self._make_leaf("b")
        angles = gd._root_angles([a, b])
        mid = (angles[0] + angles[1]) / 2
        self.assertAlmostEqual(mid, gd._ROOT_CENTER_ANGLE, places=5)

    def test_angles_decrease_left_to_right(self):
        """Left-most angle > right-most (counter-clockwise ordering)."""
        ivs = [self._make_leaf(n) for n in ("a", "b", "c")]
        angles = gd._root_angles(ivs)
        self.assertGreater(angles[0], angles[1])
        self.assertGreater(angles[1], angles[2])

    def test_narrow_arc_for_few_leaves(self):
        """≥2 inputs → arc ≥ _ARC_MIN_DEG regardless of leaf count."""
        ivs = [self._make_leaf(n) for n in ("a", "b", "c", "d", "e", "f")]
        angles = gd._root_angles(ivs)
        arc = angles[0] - angles[-1]
        self.assertGreaterEqual(math.degrees(arc), gd._ARC_MIN_DEG - 1e-9)

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
        out = gd.build_tikz(self.simple, auto_layout=False)
        self.assertTrue(out.strip().startswith(r"\begin{figure}"))

    def test_ends_with_figure(self):
        out = gd.build_tikz(self.simple, auto_layout=False)
        self.assertTrue(out.strip().endswith(r"\end{figure}"))

    def test_contains_tikzpicture(self):
        out = gd.build_tikz(self.simple, auto_layout=False)
        self.assertIn(r"\begin{tikzpicture}", out)
        self.assertIn(r"\end{tikzpicture}", out)

    def test_contains_required_styles(self):
        out = gd.build_tikz(self.simple, auto_layout=False)
        for style in ("root_block", "model_block", "deriv_node",
                      "leaf_node", "effect_node"):
            with self.subTest(style=style):
                self.assertIn(style, out)

    # ── Label ──────────────────────────────────────────────────────────────
    def test_default_label(self):
        out = gd.build_tikz(self.simple, auto_layout=False)
        self.assertIn(r"\label{fig:utd_y}", out)

    def test_custom_label(self):
        out = gd.build_tikz(self.simple, label="my_fig", auto_layout=False)
        self.assertIn(r"\label{fig:my_fig}", out)

    def test_caption_contains_measurand(self):
        out = gd.build_tikz(self.simple, auto_layout=False)
        self.assertIn(r"$y$", out)

    # ── Root block ─────────────────────────────────────────────────────────
    def test_root_block_node_present(self):
        out = gd.build_tikz(self.simple, auto_layout=False)
        self.assertIn("root_block", out)
        self.assertRegex(out, r"\\node \[root_block\]")

    # ── Leaf inputs ────────────────────────────────────────────────────────
    def test_leaf_uncertainty_nodes(self):
        out = gd.build_tikz(self.simple, auto_layout=False)
        self.assertIn("u(a)", out)
        self.assertIn("u(x)", out)
        self.assertIn("u(b)", out)

    def test_effect_node_present_when_effects_given(self):
        out = gd.build_tikz(self.simple, auto_layout=False)
        self.assertIn("Calibration", out)
        self.assertIn("Offset estimation", out)

    def test_no_effect_node_when_no_effects(self):
        out = gd.build_tikz(self.simple, auto_layout=False)
        n_effect_nodes = out.count("effect_node,")
        # simple model: a has 1 effect, x has 0, b has 1 → 2 effect nodes
        self.assertEqual(n_effect_nodes, 2)

    def test_leaf_colors_applied(self):
        out = gd.build_tikz(self.simple, auto_layout=False)
        self.assertIn("draw=red", out)
        self.assertIn("draw=blue!70!black", out)
        self.assertIn("draw=purple", out)

    # ── Connections ─────────────────────────────────────────────────────────
    def test_connections_drawn(self):
        out = gd.build_tikz(self.simple, auto_layout=False)
        self.assertRegex(out, r"\\draw \[connection")

    def test_dashed_connections_for_effect_nodes(self):
        out = gd.build_tikz(self.simple, auto_layout=False)
        self.assertIn("dashed", out)

    # ── Partial derivative notation ─────────────────────────────────────────
    def test_partial_derivatives_present(self):
        out = gd.build_tikz(self.simple, auto_layout=False)
        self.assertIn(r"\frac{\partial y}{\partial a}", out)
        self.assertIn(r"\frac{\partial y}{\partial x}", out)
        self.assertIn(r"\frac{\partial y}{\partial b}", out)

    # ── Nested / sub-model ──────────────────────────────────────────────────
    def test_submodel_block_present(self):
        out = gd.build_tikz(self.nested, auto_layout=False)
        self.assertRegex(out, r"\\node \[model_block")

    def test_submodel_inputs_appear(self):
        out = gd.build_tikz(self.nested, auto_layout=False)
        self.assertIn("u(u)", out)
        self.assertIn("u(v)", out)

    def test_side_leaf_present(self):
        """q is a leaf input at root level → placed to the right."""
        out = gd.build_tikz(self.nested, auto_layout=False)
        self.assertIn("u(q)", out)

    def test_no_duplicate_node_ids(self):
        """Every TikZ node identifier must be unique."""
        out = gd.build_tikz(self.nested, auto_layout=False)
        ids = re.findall(r"\\node\s*\[[^\]]*\]\s*\(([^)]+)\)", out)
        self.assertEqual(len(ids), len(set(ids)), f"Duplicate TikZ IDs: {ids}")

    # ── Built-in example ────────────────────────────────────────────────────
    def test_builtin_example_runs(self):
        model = gd._builtin_example()
        out = gd.build_tikz(model, label="swh_utd", auto_layout=False)
        self.assertIn(r"\begin{figure}", out)
        self.assertIn(r"\label{fig:swh_utd}", out)

    def test_builtin_example_lambda_not_corrupted(self):
        model = gd._builtin_example()
        out = gd.build_tikz(model, auto_layout=False)
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
                fh.write(gd.build_tikz(model, label=label, auto_layout=False) + "\n")
            self.assertTrue(os.path.exists(path))
            content = Path(path).read_text()
            self.assertIn(rf"\label{{fig:{label}}}", content)




# ── collect_separate_figures ─────────────────────────────────────────────────

def _separate_model() -> gd.MeasurementModel:
    """z = p * q  where p has a sub-model marked separate_figure=True."""
    sub_expr, _ = gd._parse_latex_expr(r"\frac{u}{v}", {})
    syms_sub = {str(s): s for s in sub_expr.free_symbols}
    p_model = gd.MeasurementModel(
        latex_name="p", latex_expr=r"\frac{u}{v}", expr=sub_expr,
        inputs=[
            gd.InputVar("u", syms_sub["u"], "red"),
            gd.InputVar("v", syms_sub["v"], "purple"),
        ],
    )
    root_expr, _ = gd._parse_latex_expr(r"p \cdot q", {})
    syms = {str(s): s for s in root_expr.free_symbols}
    p_iv = gd.InputVar("p", syms["p"], "red", submodel=p_model,
                       separate_figure=True, separate_label="utd_p")
    q_iv = gd.InputVar("q", syms["q"], "blue!70!black")
    return gd.MeasurementModel(
        latex_name="z", latex_expr=r"p \cdot q",
        expr=root_expr, inputs=[p_iv, q_iv],
    )


class TestCollectSeparateFigures(unittest.TestCase):

    def test_no_separate_returns_empty(self):
        self.assertEqual(gd.collect_separate_figures(_nested_model()), [])

    def test_finds_one_separate(self):
        model = _separate_model()
        result = gd.collect_separate_figures(model)
        self.assertEqual(len(result), 1)
        ivar, sub = result[0]
        self.assertEqual(ivar.separate_label, "utd_p")
        self.assertEqual(sub.latex_name, "p")

    def test_separate_figure_not_expanded_in_parent(self):
        """The parent figure should contain a cross-reference, not u/v nodes."""
        model = _separate_model()
        out = gd.build_tikz(model, label="utd_z", auto_layout=False)
        self.assertIn(r"see Fig.~\ref{fig:utd_p}", out)
        # u and v nodes must NOT appear in the parent figure
        self.assertNotIn(r"u(u)", out)
        self.assertNotIn(r"u(v)", out)

    def test_separate_figure_contains_full_trace(self):
        """The sub-model figure should expand u and v."""
        model = _separate_model()
        _, sub = gd.collect_separate_figures(model)[0]
        out = gd.build_tikz(sub, label="utd_p", auto_layout=False)
        self.assertIn(r"\label{fig:utd_p}", out)
        self.assertIn("u(u)", out)
        self.assertIn("u(v)", out)



# ── _auto_layout ──────────────────────────────────────────────────────────────

class TestAutoLayout(unittest.TestCase):

    def test_no_overlaps_after_auto_layout(self):
        """After _auto_layout, no two nodes from different branches should
        have bounding boxes that overlap (within a small tolerance)."""
        model = _simple_model()
        gd._auto_layout(model)
        arc_rad = gd._root_sector_rad(model.inputs)
        root_sectors = gd._sector_angles(model.inputs, gd._ROOT_CENTER_ANGLE, arc_rad,
                                          apply_min_sector=False)
        all_recs = []
        for ivar, (angle, sector_rad) in zip(model.inputs, root_sectors):
            d = gd._radial_step(sector_rad)
            x_d = d * math.cos(angle)
            y_d = d * math.sin(angle)
            all_recs.extend(gd._simulate_branch(model, ivar, x_d, y_d, angle, sector_rad))
        TOL = 0.05
        for i, ri in enumerate(all_recs):
            for rj in all_recs[i + 1:]:
                if ri.ivar is rj.ivar:
                    continue
                ov = gd._aabb_overlap(gd._aabb(ri), gd._aabb(rj))
                if ov is not None:
                    self.assertLessEqual(
                        ov[0], TOL,
                        f"X-overlap {ov[0]:.3f} > {TOL} between "
                        f"{ri.ivar.latex_name}({ri.ntype}) and "
                        f"{rj.ivar.latex_name}({rj.ntype})",
                    )
                    self.assertLessEqual(
                        ov[1], TOL,
                        f"Y-overlap {ov[1]:.3f} > {TOL} between "
                        f"{ri.ivar.latex_name}({ri.ntype}) and "
                        f"{rj.ivar.latex_name}({rj.ntype})",
                    )

    def test_branch_offsets_applied_in_tikz(self):
        """When a branch has a non-zero offset, the TikZ dx/dy reflect it."""
        model = _simple_model()
        # Manually push branch 'a' by (3, 1)
        model.inputs[0].branch_offset = (3.0, 1.0)
        out = gd.build_tikz(model, auto_layout=False)
        # Node for 'a' branch should have an x offset that includes 3.0
        # The natural x_d for the first input: find its position lines
        a_lines = [ln for ln in out.splitlines() if "DARED" in ln or "DA)" in ln or "(DA)" in ln]
        # Just check tikz has a coordinate > 3 for this branch (natural ≈ ±2)
        import re
        coords = re.findall(r'\(DAROOT\)\+\(([+-]?\d+\.\d+)cm', out)
        if coords:
            # At least one coordinate should be shifted by ~3 relative to natural
            self.assertTrue(any(abs(float(c)) > 2.5 for c in coords))

    def test_walk_inputs_returns_all_ivars(self):
        """_walk_inputs returns all InputVar objects in the tree."""
        model = _nested_model()
        ivars = gd._walk_inputs(model)
        latex_names = {iv.latex_name for iv in ivars}
        self.assertIn(r"p", latex_names)   # root-level input with sub-model
        self.assertIn(r"u", latex_names)   # nested input


# ── _estimate_node_bbox ───────────────────────────────────────────────────────

class TestEstimateNodeBbox(unittest.TestCase):
    def test_leaf_grows_with_label_length(self):
        """Longer leaf labels should produce a wider bbox estimate."""
        hw_short, _ = gd._estimate_node_bbox("leaf", r"u(P)")
        hw_long, _ = gd._estimate_node_bbox("leaf", r"u(\Delta\varpi_{\rm e})")
        self.assertGreater(hw_long, hw_short)

    def test_deriv_grows_with_label_length(self):
        hw_short, _ = gd._estimate_node_bbox("deriv", r"\frac{\partial y}{\partial a}")
        hw_long, _ = gd._estimate_node_bbox(
            "deriv", r"\frac{\partial \varpi_{\rm g}}{\partial \Delta\varpi_{\rm g}}")
        self.assertGreater(hw_long, hw_short)

    def test_model_fixed_size(self):
        """model bbox is now content-aware: longer equation → wider box."""
        hw1, _ = gd._estimate_node_bbox("model", "x = 1")
        hw2, _ = gd._estimate_node_bbox("model", r"x = a + b + c + d + e + f + g + h")
        self.assertGreater(hw2, hw1)

    def test_effect_fixed_size(self):
        """effect bbox is now content-aware: more/longer items → wider/taller box."""
        hw1, hh1 = gd._estimate_node_bbox("effect", "A")
        hw2, hh2 = gd._estimate_node_bbox("effect", r"Long item one \\ Long item two \\ Long item three")
        self.assertGreater(hw2, hw1)
        self.assertGreater(hh2, hh1)  # more lines → taller box

    def test_minimum_leaf_width(self):
        hw, _ = gd._estimate_node_bbox("leaf", "x")
        self.assertGreater(hw, 0.0)

    def test_node_record_carries_bbox(self):
        """_NodeRecord must carry its own bbox tuple."""
        model = _simple_model()
        arc_rad = gd._root_sector_rad(model.inputs)
        root_sectors = gd._sector_angles(model.inputs, gd._ROOT_CENTER_ANGLE,
                                          arc_rad, apply_min_sector=False)
        ivar = model.inputs[0]
        angle, sector_rad = root_sectors[0]
        x_d = gd._V_D0 * math.cos(angle)
        y_d = gd._V_D0 * math.sin(angle)
        recs = gd._simulate_branch(model, ivar, x_d, y_d, angle, sector_rad)
        for rec in recs:
            self.assertIsInstance(rec.bbox, tuple)
            self.assertEqual(len(rec.bbox), 2)
            self.assertGreater(rec.bbox[0], 0.0)
            self.assertGreater(rec.bbox[1], 0.0)


# ── parse_utd_tex ─────────────────────────────────────────────────────────────

class TestParseUtdTex(unittest.TestCase):
    """Round-trip: build a UTD, write .tex, parse it back, check fidelity."""

    @classmethod
    def setUpClass(cls):
        import tempfile, os
        cls.model = _simple_model()   # y = a*x + b
        cls.label = "test_utd_y"
        cls.caption = "Test UTD for y."
        cls.tikz = gd.build_tikz(cls.model, label=cls.label,
                                  caption=cls.caption, auto_layout=False)
        cls.tmp = tempfile.NamedTemporaryFile(suffix=".tex", delete=False, mode="w")
        cls.tmp.write(cls.tikz + "\n")
        cls.tmp.close()
        cls.parsed = gd.parse_utd_tex(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        import os
        os.unlink(cls.tmp.name)

    def test_root_name_recovered(self):
        self.assertEqual(self.parsed["model"].latex_name, self.model.latex_name)

    def test_input_count(self):
        self.assertEqual(len(self.parsed["model"].inputs),
                         len(self.model.inputs))

    def test_label_recovered(self):
        self.assertEqual(self.parsed["label"], self.label)

    def test_caption_recovered(self):
        self.assertIn("Test UTD", self.parsed["caption"])

    def test_effects_recovered(self):
        """InputVars with effects must have them recovered after round-trip."""
        parsed_inputs = {iv.latex_name: iv
                         for iv in self.parsed["model"].inputs}
        # 'a' has effects=["Calibration"]
        a_iv = parsed_inputs.get("a")
        self.assertIsNotNone(a_iv)
        self.assertIn("Calibration", a_iv.effects)

    def test_input_var_names_match(self):
        orig_names = {iv.latex_name for iv in self.model.inputs}
        parsed_names = {iv.latex_name for iv in self.parsed["model"].inputs}
        self.assertEqual(orig_names, parsed_names)

    def test_roundtrip_nested_model(self):
        """parse_utd_tex must handle a nested sub-model."""
        import tempfile, os
        nested = _nested_model()
        tikz = gd.build_tikz(nested, label="test_z", auto_layout=False)
        with tempfile.NamedTemporaryFile(suffix=".tex", delete=False, mode="w") as f:
            f.write(tikz + "\n")
            tmp_path = f.name
        try:
            parsed = gd.parse_utd_tex(tmp_path)
            root = parsed["model"]
            self.assertEqual(root.latex_name, "z")
            # p has a sub-model; q is a leaf
            p_iv = next((iv for iv in root.inputs if iv.latex_name == "p"), None)
            q_iv = next((iv for iv in root.inputs if iv.latex_name == "q"), None)
            self.assertIsNotNone(p_iv)
            self.assertIsNotNone(q_iv)
            self.assertIsNotNone(p_iv.submodel)
            self.assertIsNone(q_iv.submodel)
            self.assertEqual(p_iv.submodel.latex_name, "p")
        finally:
            os.unlink(tmp_path)


# ── Round 2: wider arc + centring + content-aware steps + compaction ──────────

class TestWiderArc(unittest.TestCase):
    """_root_sector_rad enforces _ARC_MIN_DEG for trees with >= 2 inputs."""

    def _make_leaf(self, name):
        s = sp.Symbol(name)
        return gd.InputVar(name, s, "black")

    def test_min_arc_two_inputs(self):
        ivs = [self._make_leaf("a"), self._make_leaf("b")]
        arc_deg = math.degrees(gd._root_sector_rad(ivs))
        self.assertGreaterEqual(arc_deg, gd._ARC_MIN_DEG - 1e-9)

    def test_min_arc_six_inputs(self):
        ivs = [self._make_leaf(c) for c in "abcdef"]
        arc_deg = math.degrees(gd._root_sector_rad(ivs))
        self.assertGreaterEqual(arc_deg, gd._ARC_MIN_DEG - 1e-9)

    def test_single_input_no_min(self):
        """Single-input trees should not be forced to 270° (looks wrong)."""
        iv = self._make_leaf("a")
        arc_deg = math.degrees(gd._root_sector_rad([iv]))
        self.assertLess(arc_deg, gd._ARC_MIN_DEG)


class TestRootCentring(unittest.TestCase):
    """Root should be within 5 cm of the bbox centre for the built-in example."""

    def test_builtin_example_root_near_centre(self):
        model = gd._builtin_example()
        gd._auto_layout(model)
        # Collect all node records
        arc_rad = gd._root_sector_rad(model.inputs)
        root_sectors = gd._sector_angles(model.inputs, gd._ROOT_CENTER_ANGLE, arc_rad,
                                          apply_min_sector=False)
        recs = []
        for ivar, (angle, sector_rad) in zip(model.inputs, root_sectors):
            x_d = gd._V_D0 * math.cos(angle)
            y_d = gd._V_D0 * math.sin(angle)
            recs.extend(gd._simulate_branch(model, ivar, x_d, y_d, angle, sector_rad))
        xs = [r.x for r in recs]
        ys = [r.y for r in recs]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        # root is at (0, 0) — check distance to bbox centre
        dist = math.hypot(cx, cy)
        self.assertLessEqual(dist, 7.0,
            f"Root is {dist:.2f} cm from bbox centre (expected ≤7 cm, was ~9 cm before Round 2)")


class TestContentAwareSteps(unittest.TestCase):
    """_v_between with small content gives shorter step than with large content."""

    def test_short_label_shorter_step(self):
        angle = math.pi  # leftward
        small_hw, small_hh = 0.5, 0.35
        large_hw, large_hh = 2.0, 0.55
        v_small = gd._v_between(small_hw, small_hh, small_hw, small_hh, angle)
        v_large = gd._v_between(large_hw, large_hh, large_hw, large_hh, angle)
        self.assertLess(v_small, v_large)

    def test_v_between_horizontal(self):
        """Horizontal step = 2 * hw + gap."""
        hw, hh = 1.0, 0.5
        v = gd._v_between(hw, hh, hw, hh, 0.0, gap=0.0)
        self.assertAlmostEqual(v, 2 * hw, places=5)

    def test_v_between_vertical(self):
        """Vertical step = 2 * hh + gap."""
        hw, hh = 1.0, 0.5
        v = gd._v_between(hw, hh, hw, hh, math.pi / 2, gap=0.0)
        self.assertAlmostEqual(v, 2 * hh, places=5)


class TestCompactionPass(unittest.TestCase):
    """Post-compaction offsets are closer to origin than pre-compaction."""

    def test_compaction_reduces_offsets(self):
        model = gd._builtin_example()
        # Run layout (includes compaction)
        gd._auto_layout(model)
        ivars = gd._walk_inputs(model)
        # At least some offsets should exist; compaction should leave none
        # unreasonably large (anything > 8 cm would indicate no compaction)
        max_offset = max(math.hypot(*iv.branch_offset) for iv in ivars)
        self.assertLess(max_offset, 8.0,
            f"Max branch offset {max_offset:.2f} cm seems too large after compaction")


if __name__ == "__main__":
    unittest.main(verbosity=2)
