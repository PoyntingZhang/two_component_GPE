"""Restricted Python expressions and an offline mathematical preview."""

import ast
from html import escape
import math


FUNCTIONS = {"sin", "cos", "tan", "exp", "sqrt", "tanh", "abs", "log"}
BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)


def parse_expression(source, time_dependent=False):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Enter a formula, or 0 to disable this potential.")
    if len(source) > 4096:
        raise ValueError("Keep the formula within 4096 characters.")
    try:
        tree = ast.parse(source.strip(), mode="eval").body
    except (SyntaxError, RecursionError) as exc:
        message = exc.msg if isinstance(exc, SyntaxError) else "expression is too deeply nested"
        raise ValueError(f"Syntax error: {message}.") from exc
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError("The formula is too complex; use a shorter expression.")
    variables = {"x", "y", "z", "pi"} | ({"t"} if time_dependent else set())

    def check(node):
        if isinstance(node, ast.Constant):
            try:
                finite = type(node.value) in (int, float) and math.isfinite(node.value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("Use finite real numbers in the formula.")
        elif isinstance(node, ast.Name):
            if node.id not in variables:
                if node.id in FUNCTIONS:
                    raise ValueError(f"Use {node.id}(...) to call this function.")
                raise ValueError(f"Unrecognised variable {node.id}.")
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            check(node.operand)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, BINARY):
            check(node.left)
            check(node.right)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Use function names directly, such as sin(x), without np. or cp.")
            if node.func.id not in FUNCTIONS:
                raise ValueError(f"Unrecognised function {node.func.id}.")
            if len(node.args) != 1 or node.keywords:
                raise ValueError(f"{node.func.id} expects one positional argument.")
            check(node.args[0])
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
            raise ValueError("Use ** for powers, for example x**2, rather than ^.")
        else:
            raise ValueError("Use numbers, coordinates, + - * / **, parentheses and the listed functions.")

    check(tree)
    return tree


def is_zero(node):
    while isinstance(node, ast.UnaryOp):
        node = node.operand
    return isinstance(node, ast.Constant) and node.value == 0


def mathml(node):
    def fenced(body):
        return f"<mrow><mo>(</mo>{body}<mo>)</mo></mrow>"

    def precedence(item):
        if isinstance(item, ast.BinOp):
            return 10 if isinstance(item.op, (ast.Add, ast.Sub)) else 20 if isinstance(item.op, (ast.Mult, ast.Div)) else 40
        return 30 if isinstance(item, ast.UnaryOp) else 50

    def render(item):
        if isinstance(item, ast.Constant):
            value = str(item.value)
            if "e" in value.lower():
                mantissa, exponent = value.lower().split("e")
                return f"<mrow><mn>{mantissa}</mn><mo>×</mo><msup><mn>10</mn><mn>{int(exponent)}</mn></msup></mrow>"
            return f"<mn>{value}</mn>"
        if isinstance(item, ast.Name):
            return f"<mi>{'π' if item.id == 'pi' else item.id}</mi>"
        if isinstance(item, ast.UnaryOp):
            body = render(item.operand)
            if precedence(item.operand) < 30:
                body = fenced(body)
            return f"<mrow><mo>{'−' if isinstance(item.op, ast.USub) else '+'}</mo>{body}</mrow>"
        if isinstance(item, ast.Call):
            name, argument = item.func.id, render(item.args[0])
            if name == "sqrt":
                return f"<msqrt>{argument}</msqrt>"
            if name == "abs":
                return f"<mrow><mo>|</mo>{argument}<mo>|</mo></mrow>"
            return f'<mrow><mi mathvariant="normal">{name}</mi><mo>⁡</mo>{fenced(argument)}</mrow>'
        left, right = render(item.left), render(item.right)
        if isinstance(item.op, ast.Div):
            return f"<mfrac>{left}{right}</mfrac>"
        if isinstance(item.op, ast.Pow):
            if precedence(item.left) <= 40:
                left = fenced(left)
            return f"<msup>{left}{right}</msup>"
        level = precedence(item)
        if precedence(item.left) < level:
            left = fenced(left)
        if precedence(item.right) < level or (isinstance(item.op, ast.Sub) and precedence(item.right) == level):
            right = fenced(right)
        operator = {ast.Add: "+", ast.Sub: "−", ast.Mult: "·"}[type(item.op)]
        return f"<mrow>{left}<mo>{operator}</mo>{right}</mrow>"

    return render(node)


def preview_expression(source, time_dependent=False):
    try:
        node = parse_expression(source, time_dependent)
    except ValueError as exc:
        return {"valid": False, "message": str(exc)}
    return {"valid": True, "is_zero": is_zero(node),
            "mathml": '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">'
                      f'<semantics>{mathml(node)}<annotation encoding="text/plain">'
                      f'{escape(source)}</annotation></semantics></math>'}
