"""HYBRID (neuro-symbolic) pipeline:

    problem text --(LLM formalizer)--> IR --(validation)--> Z3 solver --(presentation)--> answer

This is the main thing your team builds. The plumbing is here; the parts marked TODO(team) are yours.
Suggested order of work:
    1. write FORMALIZER_PROMPT and get formalize() returning sensible IR for one simple problem;
    2. implement validate_ir();
    3. add ONE retry with feedback in solve();
    4. fix run_solver() so that "unique" and "multiple" can be told apart;
    5. (extension) replace the LLM-based present() with deterministic code for some problem types.

See docs/ir.md for the IR format.
"""
import json
from typing import Any, Optional

from pydantic import BaseModel, Field

import ir_solver
import llm
from answer import Answer


class HybridError(Exception):
    """The hybrid pipeline could not produce an answer (recorded by run.py as an error, not as status "none")."""


class IRModel(BaseModel):
    """JSON schema the formalizer must follow. Keep it in sync with docs/ir.md."""
    variables: dict[str, Any]
    all_different: list[list[str]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    objective: Optional[dict[str, str]] = None


# Compact IR description for the model. You may edit it, but keep it consistent with ir_solver.py.
IR_SPEC = """IR format (JSON):
{
  "variables": {"<name>": [lo, hi]  or  {"values": [v1, v2, ...]}},     integer variables and their domains
  "all_different": [["<name>", ...], ...],                                 groups whose values must all differ
  "constraints": ["<expression>", ...],                                    every expression must be true
  "objective": {"maximize": "<expression>"}  or  {"minimize": "<expression>"}   ONLY for optimisation problems
}
Expressions may use: integer numbers, variable names, + - * // %, == != < <= > >=, and, or, not, abs(), parentheses.
Inside arithmetic a comparison counts as 1 (true) or 0 (false), e.g. "(A == 1) + (B == 1) == 1".
Variable names must be identifiers: letters, digits and underscores only, no spaces."""

SYSTEM = "You translate logic puzzles into a formal constraint representation. You answer only with JSON."

FORMALIZER_PROMPT = """Translate the logic puzzle below into a valid JSON IR for a Z3 integer constraint solver.
Do not solve the puzzle yourself; only output the JSON object matching the schema.

{ir_spec}

Encoding rules:
- Use valid Python/JSON identifiers (letters, digits, underscores, starting with a letter or underscore; no spaces or hyphens).
- When a specific discrete list of numbers/years/values is given in the puzzle, prefer {{"values": [...]}} with exact integers (never booleans true/false). When a continuous/integer range is given, use [lo, hi] with lo <= hi.
- Put groups of variables that must all take distinct values into a single list inside "all_different" (do not create single-element lists).
- Write each condition ONCE in "constraints" without repeating or duplicating similar lines. Use only allowed syntax (+ - * // %, == != < <= > >=, and, or, not, abs(), parentheses).
- For "adjacent" / "next to" / "immediately next to" on a line/grid/floors: use "abs(X - Y) == 1".
- For "X is immediately to the left of Y" or "directly below Y": use "Y - X == 1" (or "X + 1 == Y").
- For "X is somewhere to the left of Y" or "on a lower floor than Y" or "before Y": use "X < Y".
- For "X is N years/units after Y": use "X - Y == N" (or "X == Y + N").
- For knights and knaves (truth-tellers = 1, liars = 0): if person X in [0, 1] says statement S, encode "(X == 1) == (S)".
- Include "objective" ({{"maximize": "<expr>"}} or {{"minimize": "<expr>"}}) ONLY when optimisation is asked.
- Keep the JSON concise and direct; do not repeat constraints.

Example (not from any puzzle set):
Puzzle: "Three boxes A, B, C have integer weights from 1 to 5. A and B have different weights. A is lighter than B, and B is immediately lighter than C."
IR:
{{"variables": {{"A": [1, 5], "B": [1, 5], "C": [1, 5]}}, "all_different": [["A", "B"]], "constraints": ["A < B", "C - B == 1"]}}

Puzzle:
{text}
{feedback}"""


def formalize(problem: dict, feedback: Optional[str] = None) -> dict:
    """Ask the model for an IR encoding of the problem. `feedback` = error messages from a previous attempt."""
    fb = f"\nYour previous encoding was rejected:\n{feedback}\nFix these problems.\n" if feedback else ""
    prompt = FORMALIZER_PROMPT.format(ir_spec=IR_SPEC, text=problem["text"], feedback=fb)
    parsed, raw = llm.ask_json(prompt, IRModel, system=SYSTEM)
    return parsed.model_dump(exclude_none=True)


def _is_strict_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def validate_ir(ir: dict) -> list:
    """Return a list of human-readable error messages; an empty list means the IR is valid."""
    errors: list[str] = []

    if not isinstance(ir, dict):
        return ["IR must be a JSON object."]

    variables = ir.get("variables")
    if not isinstance(variables, dict) or not variables:
        return ["'variables' must be a non-empty dictionary mapping variable names to domains."]

    for name, dom in variables.items():
        if not isinstance(name, str) or not name.isidentifier():
            errors.append(f"Invalid variable name '{name}': must be a valid identifier without spaces or special chars.")
            continue

        if isinstance(dom, list):
            if len(dom) != 2 or not _is_strict_int(dom[0]) or not _is_strict_int(dom[1]):
                errors.append(f"Domain for '{name}' must be a list of two integers [lo, hi].")
            elif dom[0] > dom[1]:
                errors.append(f"Invalid domain bounds for '{name}': lower bound {dom[0]} is greater than upper bound {dom[1]}.")
        elif isinstance(dom, dict):
            if set(dom.keys()) != {"values"} or not isinstance(dom["values"], list) or not dom["values"]:
                errors.append(f"Domain object for '{name}' must contain a non-empty list 'values'.")
            elif not all(_is_strict_int(v) for v in dom["values"]):
                errors.append(f"All elements in '{name}.values' must be integers.")
        else:
            errors.append(f"Invalid domain for '{name}': expected [lo, hi] or {{\"values\": [...]}}.")

    all_diff = ir.get("all_different", [])
    if all_diff is not None:
        if not isinstance(all_diff, list):
            errors.append("'all_different' must be a list of variable name lists.")
        else:
            for idx, group in enumerate(all_diff):
                if not isinstance(group, list) or not all(isinstance(v, str) for v in group):
                    errors.append(f"'all_different[{idx}]' must be a list of variable name strings.")
                    continue
                for var_name in group:
                    if var_name not in variables:
                        errors.append(f"Unknown variable '{var_name}' in all_different[{idx}].")

    constraints = ir.get("constraints", [])
    if constraints is not None:
        if not isinstance(constraints, list) or not all(isinstance(c, str) for c in constraints):
            errors.append("'constraints' must be a list of constraint strings.")

    objective = ir.get("objective")
    if objective is not None:
        if not isinstance(objective, dict) or len(objective) != 1:
            errors.append("'objective' must be a single-key dict with 'maximize' or 'minimize'.")
        else:
            sense, expr = next(iter(objective.items()))
            if sense not in ("maximize", "minimize") or not isinstance(expr, str) or not expr.strip():
                errors.append("'objective' must map 'maximize' or 'minimize' to a non-empty expression string.")

    if isinstance(all_diff, list):
        for idx, group in enumerate(all_diff):
            if isinstance(group, list) and len(group) < 2:
                errors.append(f"'all_different[{idx}]' must contain at least two variables.")

    if isinstance(constraints, list) and all(isinstance(c, str) for c in constraints):
        if len(constraints) != len(set(constraints)):
            errors.append("'constraints' contains duplicate expressions; remove duplicates.")

    if not errors:
        try:
            ir_solver.build(ir)
        except Exception as exc:
            errors.append(f"IR validation failed in solver build: {exc}")

    return errors


def run_solver(ir: dict) -> dict:
    """Run Z3 on the IR. Returns {"status": ..., "solutions": [...], "objective": ...}."""
    return ir_solver.solve(ir, limit=10)


PRESENT_PROMPT = """A puzzle has been solved by a constraint solver. Rewrite the solver's result in the required
answer format. Do NOT solve the puzzle again - only translate the solver's assignment.

Puzzle:
{text}

Encoding used by the solver:
{ir}

Solver assignment (variable -> value):
{assignment}

{answer_format}"""


def present(problem: dict, ir: dict, result: dict) -> dict:
    """Turn the solver result into the required answer format."""
    status = result["status"]
    if status == "none":
        return Answer(status="none").model_dump()
    prompt = PRESENT_PROMPT.format(
        text=problem["text"],
        ir=json.dumps(ir, ensure_ascii=False),
        assignment=json.dumps(result["solutions"][0], ensure_ascii=False),
        answer_format=problem["answer_format"],
    )
    parsed, raw = llm.ask_json(prompt, Answer)
    return Answer(status=status, solution=parsed.solution, objective=result.get("objective")).model_dump()


def solve(problem: dict) -> dict:
    """Returns {"answer": <Answer as dict>, "trace": {...}}. Raise HybridError if no answer can be produced."""
    trace = {"attempts": []}

    ir = formalize(problem)
    errors = validate_ir(ir)
    trace["attempts"].append({"ir": ir, "errors": errors})

    if errors:
        retry_ir = formalize(problem, feedback="\n".join(errors))
        retry_errors = validate_ir(retry_ir)
        trace["attempts"].append({"ir": retry_ir, "errors": retry_errors})
        ir, errors = retry_ir, retry_errors

    if errors:
        raise HybridError("invalid IR: " + "; ".join(errors))

    try:
        result = run_solver(ir)
    except ValueError as e:  # the solver rejected the IR
        raise HybridError(f"solver rejected the IR: {e}") from e
    trace["solver"] = {
        "status": result["status"],
        "n_solutions": len(result["solutions"]),
        "objective": result.get("objective"),
        "first_solution": result["solutions"][0] if result["solutions"] else None,
    }

    answer = present(problem, ir, result)
    return {"answer": answer, "trace": trace}