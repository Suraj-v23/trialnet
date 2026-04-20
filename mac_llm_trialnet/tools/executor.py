"""
tools/executor.py — Tool registry and execution for TrialNet Phase 5.

Three tools:
  calculator   — safe arithmetic evaluation
  python_exec  — sandboxed Python snippet execution
  search_memory — RAG query against ChromaDB mistake bank
"""

from __future__ import annotations
import ast
import json
import math
import operator
import os
import sys
import traceback
from io import StringIO

# ── Tool schemas (Qwen function-call format) ─────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a mathematical expression and return the exact result. "
                "Use for any arithmetic, percentage, or algebraic computation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid Python math expression, e.g. '127 * 43' or '(1.10 - 1) / 2'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": (
                "Execute a Python code snippet and capture stdout. "
                "Use ONLY when asked to RUN or TEST code. "
                "Do NOT use when asked to WRITE or IMPLEMENT a function — just write the code directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use print() to produce output.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search the mistake memory bank for past errors similar to this query. "
                "Use when the question is complex and you want to avoid repeating past mistakes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic or question to search for in past mistakes.",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return (default 2).",
                        "default": 2,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ── Safe calculator ───────────────────────────────────────────────────────────

_SAFE_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_SAFE_NAMES.update({"abs": abs, "round": round, "int": int, "float": float})


def _safe_eval(expr: str) -> float | int:
    """Evaluate arithmetic expression with no access to builtins beyond math."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _SAFE_NAMES):
                raise ValueError(f"Function '{getattr(node.func, 'id', '?')}' not allowed")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Imports not allowed in calculator")
    return eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, _SAFE_NAMES)


def tool_calculator(expression: str) -> dict:
    try:
        result = _safe_eval(expression.strip())
        # Return int if result is whole number
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e), "expression": expression}


# ── Python exec (sandboxed stdout capture) ───────────────────────────────────

_EXEC_TIMEOUT_LINES = 200   # cap output lines

def tool_python_exec(code: str) -> dict:
    buf = StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = buf
        sys.stderr = buf
        # Restricted globals — no file/network/os access
        _builtins_dict = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
        safe_globals = {
            "__builtins__": {
                k: _builtins_dict[k]
                for k in ["print", "range", "len", "int", "float", "str", "list",
                          "dict", "set", "tuple", "bool", "enumerate", "zip",
                          "map", "filter", "sorted", "sum", "min", "max", "abs",
                          "round", "type", "isinstance", "repr", "__import__",
                          "__build_class__"]
                if k in _builtins_dict
            }
        }
        # Allow importing from project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        exec(code, safe_globals)
        output = buf.getvalue()
        lines = output.splitlines()
        if len(lines) > _EXEC_TIMEOUT_LINES:
            output = "\n".join(lines[:_EXEC_TIMEOUT_LINES]) + "\n... (truncated)"
        return {"output": output.strip()}
    except Exception:
        return {"error": traceback.format_exc(limit=3).strip()}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ── Memory search ─────────────────────────────────────────────────────────────

def tool_search_memory(query: str, n: int = 2) -> dict:
    try:
        from memory.chroma_bank import ChromaMemoryBank
        bank = ChromaMemoryBank()
        mistakes = bank.query_similar(query, n=n)
        if not mistakes:
            return {"results": [], "message": "No similar past mistakes found."}
        results = []
        for m in mistakes:
            results.append({
                "prompt":      m.get("prompt", "")[:120],
                "wrong":       m.get("bad_generation", "")[:80],
                "correction":  m.get("human_correction", "")[:120],
            })
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}


# ── Dispatcher ────────────────────────────────────────────────────────────────

def execute_tool(name: str, arguments: dict) -> str:
    """Execute a named tool with arguments dict. Returns JSON string."""
    if name == "calculator":
        result = tool_calculator(arguments.get("expression", ""))
    elif name == "python_exec":
        result = tool_python_exec(arguments.get("code", ""))
    elif name == "search_memory":
        result = tool_search_memory(
            arguments.get("query", ""),
            n=int(arguments.get("n", 2)),
        )
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result)
