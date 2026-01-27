"""Typed evaluator for MultiLeaderIntermarketCondition (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import MultiLeaderIntermarketCondition

from .context import EvalContext


def evaluate_multi_leader_intermarket(
    condition: MultiLeaderIntermarketCondition, bar: Any, ctx: EvalContext
) -> bool:
    """Evaluate multi-leader intermarket: aggregate feature across leaders vs threshold."""
    if not condition.leader_symbols:
        return False

    feature_values = []
    for leader in condition.leader_symbols:
        roc_id = f"roc_{leader}_{condition.window_bars}"
        roc_ind = ctx.indicators.get(roc_id)
        if roc_ind:
            if condition.aggregate_feature == "ret_pct":
                feature_values.append(roc_ind.Current.Value * 100)
            else:
                feature_values.append(roc_ind.Current.Value)

    if not feature_values:
        return False

    if condition.aggregate_op == "max":
        aggregated = max(feature_values)
    elif condition.aggregate_op == "min":
        aggregated = min(feature_values)
    else:
        aggregated = sum(feature_values) / len(feature_values)

    return aggregated > condition.trigger_threshold
