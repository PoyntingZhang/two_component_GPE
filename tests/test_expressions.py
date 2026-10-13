from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from config.expressions import preview_expression


class ExpressionTests(unittest.TestCase):
    def test_inline_coefficients_coordinates_and_functions(self):
        for formula in ("0.03*x + 0.002*y**2", "sqrt(x**2+y**2)/2", "sin(pi*x) + cos(y)",
                        "exp(-z**2) + tanh(z) + abs(x) + log(2) + tan(y)", "1e-6*x"):
            with self.subTest(formula=formula):
                result = preview_expression(formula)
                self.assertTrue(result["valid"], result)
                ET.fromstring(result["mathml"])
                self.assertFalse(result["is_zero"])

    def test_time_is_only_available_to_dynamic_expression(self):
        formula = "0.015*(sin(0.065*t)*x+cos(0.065*t)*y)"
        self.assertTrue(preview_expression(formula, True)["valid"])
        self.assertEqual(preview_expression(formula)["message"], "Unrecognised variable t.")

    def test_unknown_names_and_function_arity_are_explained(self):
        cases = {"A*y": "Unrecognised variable A", "sinn(x)": "Unrecognised function sinn",
                 "sin(x,y)": "one positional argument", "cos": "Use cos(...)",
                 "x^2": "Use ** for powers", "np.sin(x)": "without np. or cp."}
        for formula, message in cases.items():
            with self.subTest(formula=formula):
                result = preview_expression(formula, True)
                self.assertFalse(result["valid"])
                self.assertIn(message, result["message"])

    def test_syntax_errors_remain_invalid(self):
        for formula in ("sin(", "0.03*", "x y", "x=2", "x; y"):
            with self.subTest(formula=formula):
                result = preview_expression(formula)
                self.assertFalse(result["valid"])
                self.assertTrue(result["message"].startswith("Syntax error:"))

    def test_no_python_execution_or_nonreal_values(self):
        for formula in ("__import__('os').system('echo unsafe')", "x.__class__", "[x for x in y]",
                        "(lambda: 1)()", "x[0]", "True", "1j", "1e999", "sin(x=1)",
                        '"<script>alert(1)</script>"', "x if y else z", "", " "):
            with self.subTest(formula=formula):
                result = preview_expression(formula, True)
                self.assertFalse(result["valid"])
                self.assertNotIn("mathml", result)

    def test_zero_detection_does_not_disable_nonzero_formula(self):
        for formula in ("0", "0.0", "(-0.0)", "+0", "0e10"):
            self.assertTrue(preview_expression(formula)["is_zero"])
        for formula in ("0.03*x", "x-x", "sin(0)"):
            self.assertFalse(preview_expression(formula)["is_zero"])

    def test_math_notation_preserves_grouping(self):
        self.assertIn("<mfrac>", preview_expression("x/y")["mathml"])
        self.assertIn("<msqrt>", preview_expression("sqrt(x)")["mathml"])
        self.assertIn("<msup><mi>x</mi><mn>2</mn></msup>", preview_expression("x**2")["mathml"])
        self.assertIn("<mo>(</mo>", preview_expression("(-x)**2")["mathml"])
        self.assertNotEqual(preview_expression("(-x)**2")["mathml"], preview_expression("-x**2")["mathml"])
        self.assertIn("<mo>(</mo>", preview_expression("x-(y+z)")["mathml"])
        result = preview_expression("x # <img src=x onerror=alert(1)>")
        self.assertNotIn("<img", result["mathml"])
        ET.fromstring(result["mathml"])


if __name__ == "__main__":
    unittest.main()
