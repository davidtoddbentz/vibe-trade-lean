"""Typed evaluator for SpreadCondition (Phase 7).

Multi-symbol spread: placeholder implementation when only single-symbol data.
"""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import SpreadCondition

from .context import EvalContext


def evaluate_spread(condition: SpreadCondition, bar: Any, ctx: EvalContext) -> bool:
    """Evaluate spread condition (placeholder for single-symbol: same price for both)."""
    # Placeholder: multi-symbol requires LEAN multi-symbol data
    price_a = bar.Close
    price_b = bar.Close

    if price_b == 0:
        return False

    if condition.calc_type == "ratio":
        spread_val = price_a / price_b
    elif condition.calc_type == "difference":
        spread_val = price_a - price_b
    elif condition.calc_type == "log_ratio":
        import math
        spread_val = math.log(price_a / price_b) if price_a > 0 and price_b > 0 else 0
    else:
        spread_val = 0.0

    if condition.trigger_op == "above":
        return spread_val > condition.threshold
    if condition.trigger_op == "below":
        return spread_val < condition.threshold
    if condition.trigger_op == "crosses_above":
        return spread_val > condition.threshold
    if condition.trigger_op == "crosses_below":
        return spread_val < condition.threshold
    return False
