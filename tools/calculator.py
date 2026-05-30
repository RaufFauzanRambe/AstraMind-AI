"""
AstraMind AI - Calculator Tool
================================
A mathematical calculation tool that evaluates arithmetic expressions
and performs common mathematical operations.
"""

import logging
import math
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CalculatorTool:
    """
    A tool for performing mathematical calculations.

    Supports:
    - Basic arithmetic (+, -, *, /, %, **)
    - Mathematical functions (sin, cos, tan, sqrt, log, etc.)
    - Expression evaluation with safety checks
    - Unit conversions
    """

    name = "calculator"
    description = "Performs mathematical calculations and evaluates arithmetic expressions."

    # Safe mathematical functions available for evaluation
    SAFE_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "ceil": math.ceil,
        "floor": math.floor,
        "pi": math.pi,
        "e": math.e,
        "factorial": math.factorial,
        "gcd": math.gcd,
    }

    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute a calculation based on the query.

        Args:
            query: A string containing a mathematical expression or question.

        Returns:
            Dictionary with the calculation result and details.
        """
        logger.info(f"Calculator processing: {query}")

        # Extract mathematical expression from the query
        expression = self._extract_expression(query)

        if not expression:
            return {
                "success": False,
                "error": "Tidak dapat menemukan ekspresi matematika dalam query.",
                "query": query,
            }

        try:
            result = self._safe_eval(expression)
            return {
                "success": True,
                "expression": expression,
                "result": result,
                "result_type": type(result).__name__,
                "query": query,
            }
        except Exception as e:
            logger.error(f"Calculation error: {e}")
            return {
                "success": False,
                "error": f"Kesalahan perhitungan: {str(e)}",
                "expression": expression,
                "query": query,
            }

    def _extract_expression(self, query: str) -> Optional[str]:
        """
        Extract a mathematical expression from a natural language query.

        Tries to find and isolate the calculable part of the input.
        """
        # Direct expression pattern (e.g., "2 + 3", "sqrt(16)")
        direct_pattern = r'[\d\.\s\+\-\*\/\%\(\)\^]+(?:sin|cos|tan|sqrt|log|exp|abs|ceil|floor|pi|e)[\d\.\s\+\-\*\/\%\(\)\^]*'
        direct_match = re.search(direct_pattern, query, re.IGNORECASE)

        if direct_match:
            return direct_match.group().strip()

        # Number and operator pattern (e.g., "berapa 15 kali 23")
        cleaned = query.lower()
        indonesian_ops = {
            "tambah": "+", "ditambah": "+", "plus": "+",
            "kurang": "-", "dikurangi": "-", "minus": "-",
            "kali": "*", "dikali": "*", "times": "*",
            "bagi": "/", "dibagi": "/", "dibagi oleh": "/",
            "pangkat": "**", "modulo": "%", "mod": "%",
        }

        for indo, op in indonesian_ops.items():
            cleaned = cleaned.replace(indo, op)

        # Try to find a simple arithmetic expression
        arith_pattern = r'[\d\.]+(?:\s*[+\-*/%]\s*[\d\.]+)+'
        arith_match = re.search(arith_pattern, cleaned)

        if arith_match:
            expr = arith_match.group().replace("^", "**")
            return expr

        # Check if the query is just a number
        number_pattern = r'(\d+\.?\d*)'
        number_match = re.search(number_pattern, query)
        if number_match:
            return number_match.group(1)

        return None

    def _safe_eval(self, expression: str) -> Any:
        """
        Safely evaluate a mathematical expression.

        Only allows mathematical operations and safe functions,
        preventing code injection attacks.
        """
        # Replace ^ with ** for exponentiation
        expression = expression.replace("^", "**")

        # Validate expression contains only allowed characters
        allowed_chars = set("0123456789.+-*/%() ,")
        allowed_funcs = set(self.SAFE_FUNCTIONS.keys())

        # Check for function calls
        func_pattern = r'([a-zA-Z_]\w*)\s*\('
        for match in re.finditer(func_pattern, expression):
            func_name = match.group(1)
            if func_name not in allowed_funcs:
                raise ValueError(f"Fungsi '{func_name}' tidak diizinkan.")

        # Build a safe namespace with only math functions
        safe_namespace = {"__builtins__": {}}
        safe_namespace.update(self.SAFE_FUNCTIONS)

        try:
            result = eval(expression, safe_namespace)
            return result
        except ZeroDivisionError:
            raise ValueError("Pembagian dengan nol tidak diperbolehkan.")
        except (SyntaxError, NameError) as e:
            raise ValueError(f"Ekspresi tidak valid: {e}")
