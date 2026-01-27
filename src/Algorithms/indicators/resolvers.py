"""Typed value resolution for ValueRef (from vibe-trade-shared).

Resolves a Pydantic ValueRef to a float using exhaustive match on the union.
No .get() on ref — only typed attribute access.
"""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import (
    IRExpression,
    IndicatorBandRef,
    IndicatorPropertyRef,
    IndicatorRef,
    LiteralRef,
    PriceRef,
    RollingWindowRef,
    StateRef,
    TimeRef,
    ValueRef,
    VolumeRef,
)
from vibe_trade_shared.models.ir.enums import IndicatorProperty, PriceField

from .registry import IndicatorCategory, resolve_indicator_value


def resolve_value(
    ref: ValueRef,
    bar: Any,
    *,
    indicator_registry: dict[str, Any],
    state: dict[str, Any],
    current_time: Any,
    rolling_windows: dict[str, Any] | None = None,
) -> float:
    """Resolve a typed ValueRef to a float.

    Uses exhaustive match on ref; only typed attribute access (no .get() on ref).
    """
    match ref:
        case LiteralRef():
            return ref.value

        case PriceRef():
            if ref.field == PriceField.OPEN:
                return float(bar.Open)
            if ref.field == PriceField.HIGH:
                return float(bar.High)
            if ref.field == PriceField.LOW:
                return float(bar.Low)
            # CLOSE or default
            return float(bar.Close)

        case VolumeRef():
            return float(bar.Volume)

        case TimeRef():
            if ref.component == "hour":
                return float(current_time.hour)
            if ref.component == "minute":
                return float(current_time.minute)
            if ref.component == "day_of_week":
                return float(current_time.weekday())
            return 0.0

        case StateRef():
            val = state.get(ref.state_id)
            if val is None:
                return 0.0
            return float(val)

        case IndicatorRef():
            ind_id = ref.indicator_id
            if ind_id is None:
                return 0.0
            registry_entry = indicator_registry.get(ind_id)
            if registry_entry is None:
                return 0.0
            category, indicator_or_data = registry_entry
            field = ref.field if ref.field != "value" else None
            return resolve_indicator_value(category, indicator_or_data, field)

        case IndicatorBandRef():
            registry_entry = indicator_registry.get(ref.indicator_id)
            if registry_entry is None:
                return 0.0
            category, ind = registry_entry
            if category != IndicatorCategory.LEAN or ind is None:
                return 0.0
            if ref.band == "upper":
                return ind.UpperBand.Current.Value
            if ref.band == "middle":
                return ind.MiddleBand.Current.Value
            if ref.band == "lower":
                return ind.LowerBand.Current.Value
            return 0.0

        case IndicatorPropertyRef():
            registry_entry = indicator_registry.get(ref.indicator_id)
            if registry_entry is None:
                return 0.0
            category, ind = registry_entry
            if category != IndicatorCategory.LEAN or ind is None:
                return 0.0
            prop = ref.property
            if prop == IndicatorProperty.STANDARD_DEVIATION:
                return ind.StandardDeviation.Current.Value
            if prop == IndicatorProperty.BAND_WIDTH:
                if hasattr(ind, "BandWidth"):
                    return ind.BandWidth.Current.Value
                upper = ind.UpperBand.Current.Value
                lower = ind.LowerBand.Current.Value
                middle = ind.MiddleBand.Current.Value
                if middle != 0:
                    return (upper - lower) / middle
                return 0.0
            return 0.0

        case RollingWindowRef():
            if rolling_windows is None:
                return 0.0
            rw_data = rolling_windows.get(ref.indicator_id)
            if rw_data is None:
                return 0.0
            window = rw_data.get("window") if isinstance(rw_data, dict) else None
            if window is None or not getattr(window, "IsReady", False):
                return 0.0
            count = getattr(window, "Count", 0)
            if count <= ref.offset:
                return 0.0
            return float(window[ref.offset])

        case IRExpression():
            left = resolve_value(
                ref.left,
                bar,
                indicator_registry=indicator_registry,
                state=state,
                current_time=current_time,
                rolling_windows=rolling_windows,
            )
            right = resolve_value(
                ref.right,
                bar,
                indicator_registry=indicator_registry,
                state=state,
                current_time=current_time,
                rolling_windows=rolling_windows,
            )
            if ref.op == "+":
                return left + right
            if ref.op == "-":
                return left - right
            if ref.op == "*":
                return left * right
            if ref.op == "/":
                if right == 0:
                    return 0.0
                return left / right
            return 0.0

        case _:
            return 0.0
