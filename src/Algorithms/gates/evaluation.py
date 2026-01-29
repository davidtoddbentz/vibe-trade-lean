"""Gate condition evaluation logic.

Phase 12: Extracted from StrategyRuntime.
"""

from __future__ import annotations

from typing import Any, Callable

from vibe_trade_shared.models.ir import GateRule, Condition


def evaluate_gates(
    gates: list[GateRule],
    evaluate_condition_func: Callable[[Condition, Any], bool],
    bar: Any,
) -> bool:
    """Evaluate gate conditions. Returns True if all gates pass.

    Args:
        gates: List of GateRule from IR
        evaluate_condition_func: Function to evaluate conditions
        bar: Current bar data

    Returns:
        True if all gates pass (allow mode) or no gates block (block mode)
    """
    for gate in gates:
        condition = gate.condition
        mode = gate.mode or "allow"

        result = evaluate_condition_func(condition, bar)

        if mode == "allow" and not result:
            return False
        elif mode == "block" and result:
            return False

    return True
