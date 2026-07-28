"""Stage 10 proof gate: `parse_command` genuinely rejects malformed input, never silently guesses.

Pure parser check -- no env, no model, no `--seed` flag, run once (not per-seed). Covers every
malformed-input category the plan names: wrong argument count, non-numeric coordinates, an
unknown verb, an unknown direction name, an empty string, and a handful of genuinely
ambiguous-sounding natural-language sentences a user might actually type. A few deliberately
*valid* strings (including case-insensitive verbs/directions) are mixed in as a control -- if
the parser started rejecting those too, that would be its own kind of bug (over-rejection),
not caught by a malformed-only test suite.

For every case, this script asserts the parser's actual behavior (raise vs. not) matches the
expected behavior, and -- for every expected-raise case -- that the raised message is a
`CommandParseError` (never a bare `ValueError` subclass mismatch or an unrelated exception
type) with non-empty, specific text. Results are dumped to
`runs/malformed_input_check.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

from lang_goal_rl.command_grammar import CommandParseError, parse_command

EXPERIMENT_DIR = Path(__file__).resolve().parent

MALFORMED_CASES: list[tuple[str, str]] = [
    ("goto 1.3 0.7", "wrong arg count: only 2 numbers, needs 3"),
    ("goto 1.3 0.7 0.5 0.2", "wrong arg count: 4 numbers, needs exactly 3"),
    ("goto a b c", "non-numeric coordinates"),
    ("goto 1.3 x 0.5", "one non-numeric coordinate among otherwise-valid ones"),
    ("fly 1.3 0.7 0.5", "unknown verb"),
    ("go 1.3 0.7 0.5", "unknown verb (close to 'goto' but not it)"),
    ("move left 0.05", "unknown direction: bare 'left', not the full 'reach left' phrase"),
    ("move up 0.05", "unknown direction: bare 'up'"),
    ("move reach forward", "missing distance argument"),
    ("move reach forward abc", "non-numeric distance"),
    ("move reach sideways 0.05", "unknown direction phrase entirely"),
    ("waypoints", "no legs at all"),
    ("waypoints 1.3 0.7", "one leg with only 2 numbers"),
    ("waypoints 1.3 0.7 0.5, 1.4 0.8", "second leg malformed (only 2 numbers)"),
    ("stop now", "'stop' takes no arguments"),
    ("reset please", "'reset' takes no arguments"),
    ("", "empty string"),
    ("   ", "whitespace-only string"),
    ("go somewhere nice", "ambiguous natural language, unknown verb 'go'"),
    ("please move the arm a little bit", "ambiguous natural language, unknown verb 'please'"),
    ("can you reach forward a tiny bit", "ambiguous natural language, unknown verb 'can'"),
    ("move it closer", "ambiguous natural language, looks command-like but wrong shape"),
    ("reach forward", "bare direction phrase with no 'move' verb -- 'reach' is not a known verb"),
]
"""Every case the plan explicitly names (wrong arg count, non-numeric coordinates, unknown
verb, unknown direction, empty string, ambiguous natural language) plus a few extra
edge cases exercising the same categories from a different angle."""

VALID_CONTROL_CASES: list[str] = [
    "goto 1.3 0.7 0.5",
    "GOTO 1.3 0.7 0.5",
    "move reach left 0.05",
    "MOVE REACH LEFT 0.05",
    "move reach forward -0.05",
    "waypoints 1.3 0.7 0.5, 1.4 0.8 0.6",
    "stop",
    "  stop  ",
    "reset",
]
"""Deliberately valid strings, including case-insensitive verb/direction matching and a
signed (negative) move distance -- confirms the parser isn't over-rejecting. A malformed-only
test suite would never catch a regression that made the parser too strict."""


def check_malformed_cases() -> list[dict]:
    """Run every `MALFORMED_CASES` entry through `parse_command` and record the actual outcome.

    Returns:
        One dict per case: the input text, the expected-failure reason, whether
        `CommandParseError` was actually raised, the raised message (or `None`), and whether
        this case passed (raised `CommandParseError` with a non-empty message).
    """
    results = []
    for text, reason in MALFORMED_CASES:
        try:
            command = parse_command(text)
        except CommandParseError as error:
            message = str(error)
            passed = len(message.strip()) > 0
            results.append(
                {
                    "input": text,
                    "expected_failure_reason": reason,
                    "raised_command_parse_error": True,
                    "raised_message": message,
                    "passed": passed,
                }
            )
        except Exception as error:  # noqa: BLE001 -- catching the wrong exception type is exactly the failure mode this check exists to catch
            results.append(
                {
                    "input": text,
                    "expected_failure_reason": reason,
                    "raised_command_parse_error": False,
                    "raised_message": f"WRONG EXCEPTION TYPE: {type(error).__name__}: {error}",
                    "passed": False,
                }
            )
        else:
            results.append(
                {
                    "input": text,
                    "expected_failure_reason": reason,
                    "raised_command_parse_error": False,
                    "raised_message": None,
                    "passed": False,
                    "note": f"did NOT raise -- silently parsed as {type(command).__name__}({command!r})",
                }
            )
    return results


def check_valid_control_cases() -> list[dict]:
    """Run every `VALID_CONTROL_CASES` entry and confirm it does NOT raise.

    Returns:
        One dict per case: the input text, whether it raised (should be `False`), and the
        parsed command's repr if it didn't.
    """
    results = []
    for text in VALID_CONTROL_CASES:
        try:
            command = parse_command(text)
        except CommandParseError as error:
            results.append({"input": text, "raised": True, "error": str(error), "passed": False})
        else:
            results.append({"input": text, "raised": False, "parsed": repr(command), "passed": True})
    return results


def main() -> None:
    """Run both malformed and valid-control checks, print a summary, and dump results to JSON."""
    malformed_results = check_malformed_cases()
    valid_results = check_valid_control_cases()

    n_malformed_passed = sum(1 for r in malformed_results if r["passed"])
    n_valid_passed = sum(1 for r in valid_results if r["passed"])

    print(f"malformed_cases: {n_malformed_passed}/{len(malformed_results)} correctly rejected with a clear message")
    for result in malformed_results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {result['input']!r} ({result['expected_failure_reason']}) -> {result['raised_message']}")

    print(f"valid_control_cases: {n_valid_passed}/{len(valid_results)} correctly accepted (not rejected)")
    for result in valid_results:
        status = "PASS" if result["passed"] else "FAIL"
        detail = result.get("parsed", result.get("error"))
        print(f"  [{status}] {result['input']!r} -> {detail}")

    output = {
        "malformed_cases": malformed_results,
        "valid_control_cases": valid_results,
        "n_malformed_passed": n_malformed_passed,
        "n_malformed_total": len(malformed_results),
        "n_valid_passed": n_valid_passed,
        "n_valid_total": len(valid_results),
    }
    results_path = EXPERIMENT_DIR / "runs" / "malformed_input_check.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2))
    print(f"results_saved={results_path}")


if __name__ == "__main__":
    main()
