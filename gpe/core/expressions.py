"""Translate the shared configuration expression language into CUDA arithmetic."""

import ast
import math

from config.expressions import parse_expression


def cuda_expression(source, time_dependent=False):
    tree = parse_expression(source, time_dependent=time_dependent)

    def render(node):
        if isinstance(node, ast.Constant):
            # Emit doubles so that, for example, 1/2 does not become integer division.
            return repr(float(node.value))
        if isinstance(node, ast.Name):
            return repr(math.pi) if node.id == "pi" else node.id
        if isinstance(node, ast.UnaryOp):
            sign = "-" if isinstance(node.op, ast.USub) else "+"
            return f"({sign}{render(node.operand)})"
        if isinstance(node, ast.Call):
            name = "fabs" if node.func.id == "abs" else node.func.id
            return f"{name}({render(node.args[0])})"
        left, right = render(node.left), render(node.right)
        if isinstance(node.op, ast.Pow):
            return f"pow({left}, {right})"
        operator = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[type(node.op)]
        return f"({left} {operator} {right})"

    return render(tree)
