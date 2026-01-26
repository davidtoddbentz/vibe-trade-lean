"""Condition evaluator registry for StrategyRuntime.

This module provides a registry pattern for evaluating conditions,
replacing the large switch statement in StrategyRuntime._evaluate_condition().
"""

from .registry import (
    CompareOp,
    evaluate_condition,
    CONDITION_EVALUATORS,
)

__all__ = [
    "CompareOp",
    "evaluate_condition",
    "CONDITION_EVALUATORS",
]
