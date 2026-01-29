"""Position sizing logic: scale-in, overlay scaling, accumulation checks.

Phase 12: Extracted from StrategyRuntime.
"""

from __future__ import annotations

from typing import Any, Callable

from vibe_trade_shared.models.ir import SetHoldingsAction, EntryRule, OverlayRule, Condition
from execution.types import Lot


def can_accumulate(
    entry_rule: EntryRule | None,
    current_lots: list[Lot],
    bar_count: int,
    last_entry_bar: int | None,
    entries_today: int = 0,
) -> bool:
    """Check if position policy allows another entry while invested.

    Returns True if:
    - mode is "accumulate" or "scale_in"
    - max_positions limit not reached
    - min_bars_between cooldown has passed
    - max_entries_per_day limit not reached

    Args:
        entry_rule: EntryRule from IR
        current_lots: List of open lots
        bar_count: Current bar index
        last_entry_bar: Bar index of last entry
        entries_today: Number of entries already made today

    Returns:
        True if accumulation is allowed
    """
    if not entry_rule:
        return False

    action = entry_rule.action
    # Action is typed EntryAction (SetHoldingsAction | MarketOrderAction)
    # Only SetHoldingsAction has position_policy
    if not isinstance(action, SetHoldingsAction):
        return False

    policy = action.position_policy
    if policy is None:
        return False  # No policy = single mode

    mode = policy.mode

    # Single mode: no accumulation allowed
    if mode == "single":
        return False

    # Check max_positions limit
    if policy.max_positions is not None and len(current_lots) >= policy.max_positions:
        return False

    # Check min_bars_between cooldown
    if policy.min_bars_between is not None and last_entry_bar is not None and (bar_count - last_entry_bar) < policy.min_bars_between:
        return False

    # Check max_entries_per_day limit
    if policy.max_entries_per_day is not None and entries_today >= policy.max_entries_per_day:
        return False

    return True


def apply_scale_in(
    action: SetHoldingsAction,
    current_lots: list[Lot],
    log_func: Callable[[str], None],
) -> SetHoldingsAction:
    """Apply scale-in factor if in scale_in mode with existing lots.

    Args:
        action: SetHoldingsAction to potentially modify
        current_lots: List of open lots
        log_func: Logging function

    Returns:
        Modified action (or original if no scaling applied)
    """
    if not action.position_policy:
        return action

    policy = action.position_policy
    if policy.mode != "scale_in" or not current_lots:
        return action

    scale_factor = policy.scale_factor if policy.scale_factor is not None else 0.5
    num_existing = len(current_lots)
    # Apply exponential scaling: first entry full, second * scale, third * scale^2, etc.
    scaling = scale_factor ** num_existing

    if action.sizing_mode == "pct_equity":
        original_allocation = action.allocation
        action = action.model_copy(update={"allocation": original_allocation * scaling})
        log_func(f"   Scale-in factor: {scaling:.2f} (lot #{num_existing + 1})")
    elif action.sizing_mode == "fixed_usd" and action.fixed_usd is not None:
        original_usd = action.fixed_usd
        action = action.model_copy(update={"fixed_usd": original_usd * scaling})
        log_func(f"   Scale-in: ${abs(original_usd):.2f} -> ${abs(action.fixed_usd):.2f}")

    return action


def apply_overlay_scale(
    action: SetHoldingsAction,
    overlay_scale: float,
    log_func: Callable[[str], None],
) -> SetHoldingsAction:
    """Apply overlay scaling to position size.

    Args:
        action: SetHoldingsAction to modify
        overlay_scale: Scaling factor from overlays
        log_func: Logging function

    Returns:
        Modified action (or original if scale is 1.0)
    """
    if overlay_scale == 1.0:
        return action

    if action.sizing_mode == "pct_equity":
        original_allocation = action.allocation
        action = action.model_copy(update={"allocation": original_allocation * overlay_scale})
        log_func(f"   Position scaled: {original_allocation} -> {action.allocation}")

    return action


def compute_overlay_scale(
    overlays: list[OverlayRule],
    evaluate_condition_func: Callable[[Condition, Any], bool],
    bar: Any,
    log_func: Callable[[str], None],
) -> float:
    """Compute combined overlay scaling factor for position sizing.

    Evaluates all overlays that target "entry" and multiplies their
    scale_size_frac values when conditions are true.

    Args:
        overlays: List of OverlayRule from IR
        evaluate_condition_func: Function to evaluate conditions
        bar: Current bar data
        log_func: Logging function

    Returns:
        Combined scaling factor (1.0 = no scaling)
    """
    scale = 1.0
    for overlay in overlays:
        target_roles = overlay.target_roles or ["entry", "exit"]
        if "entry" not in target_roles:
            continue

        if evaluate_condition_func(overlay.condition, bar):
            scale_size = overlay.scale_size_frac or 1.0
            scale *= scale_size
            log_func(f"   Overlay '{overlay.id or 'unknown'}' active: scale={scale_size}")

    return scale
