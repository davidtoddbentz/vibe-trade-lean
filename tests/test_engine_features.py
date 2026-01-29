"""Unit tests for engine features: notional clamping and daily entry limits.

Tests _clamp_quantity_by_notional and can_accumulate with entries_today,
without requiring the LEAN runtime.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

# Add Algorithms to path for direct imports
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src" / "Algorithms"))

from execution.actions import _clamp_quantity_by_notional  # noqa: E402
from execution.types import Lot  # noqa: E402

# Import sizing module directly to avoid position/__init__.py (LEAN dependency)
_spec = importlib.util.spec_from_file_location(
    "sizing",
    str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src" / "Algorithms" / "position" / "sizing.py"),
)
_sizing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sizing)
can_accumulate = _sizing.can_accumulate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> MagicMock:
    """Create a mock ExecutionContext with a log method."""
    ctx = MagicMock()
    ctx.log = MagicMock()
    return ctx


def _make_entry_rule(
    max_entries_per_day: int | None = None,
    mode: str = "accumulate",
    max_positions: int | None = None,
) -> MagicMock:
    """Create a mock EntryRule with a SetHoldingsAction and PositionPolicy."""
    from vibe_trade_shared.models.ir import SetHoldingsAction, PositionPolicy

    policy = PositionPolicy(
        mode=mode,
        max_positions=max_positions,
        max_entries_per_day=max_entries_per_day,
    )
    action = SetHoldingsAction(position_policy=policy)
    rule = MagicMock()
    rule.action = action
    return rule


def _make_lot() -> Lot:
    return Lot(
        lot_id=0,
        symbol="BTC-USD",
        direction="long",
        entry_time="2024-01-01T00:00:00",
        entry_price=100.0,
        entry_bar=0,
        quantity=1.0,
    )


# ===========================================================================
# _clamp_quantity_by_notional tests
# ===========================================================================


class TestClampQuantityByNotional:
    """Tests for the notional USD clamping helper."""

    def test_no_constraints_passes_through(self):
        ctx = _make_ctx()
        result = _clamp_quantity_by_notional(1.0, 100.0, None, None, ctx)
        assert result == 1.0

    def test_below_min_usd_returns_none(self):
        ctx = _make_ctx()
        # 0.5 units * $100 = $50 notional, min_usd = $100
        result = _clamp_quantity_by_notional(0.5, 100.0, 100.0, None, ctx)
        assert result is None
        ctx.log.assert_called_once()

    def test_above_max_usd_clamps(self):
        ctx = _make_ctx()
        # 2.0 units * $100 = $200 notional, max_usd = $150
        result = _clamp_quantity_by_notional(2.0, 100.0, None, 150.0, ctx)
        assert result == pytest.approx(1.5)  # 150 / 100
        ctx.log.assert_called_once()

    def test_within_range_passes_through(self):
        ctx = _make_ctx()
        # 1.0 units * $100 = $100, min=50, max=200
        result = _clamp_quantity_by_notional(1.0, 100.0, 50.0, 200.0, ctx)
        assert result == 1.0

    def test_zero_price_passes_through(self):
        ctx = _make_ctx()
        result = _clamp_quantity_by_notional(1.0, 0.0, 100.0, 200.0, ctx)
        assert result == 1.0

    def test_negative_quantity_preserves_sign(self):
        ctx = _make_ctx()
        # -2.0 units * $100 = $200 notional, max_usd = $150
        result = _clamp_quantity_by_notional(-2.0, 100.0, None, 150.0, ctx)
        assert result == pytest.approx(-1.5)  # sign preserved

    def test_min_usd_only(self):
        ctx = _make_ctx()
        # Exactly at min — should pass (not strictly less)
        result = _clamp_quantity_by_notional(1.0, 100.0, 100.0, None, ctx)
        assert result == 1.0

    def test_max_usd_only(self):
        ctx = _make_ctx()
        # Exactly at max — should pass (not strictly greater)
        result = _clamp_quantity_by_notional(1.0, 100.0, None, 100.0, ctx)
        assert result == 1.0


# ===========================================================================
# can_accumulate with entries_today tests
# ===========================================================================


class TestCanAccumulateEntriesPerDay:
    """Tests for max_entries_per_day check in can_accumulate."""

    def test_no_limit_allows_accumulation(self):
        rule = _make_entry_rule(max_entries_per_day=None)
        assert can_accumulate(rule, [_make_lot()], bar_count=10, last_entry_bar=0, entries_today=100)

    def test_under_limit_allows_entry(self):
        rule = _make_entry_rule(max_entries_per_day=5)
        assert can_accumulate(rule, [_make_lot()], bar_count=10, last_entry_bar=0, entries_today=4)

    def test_at_limit_blocks_entry(self):
        rule = _make_entry_rule(max_entries_per_day=3)
        assert not can_accumulate(rule, [_make_lot()], bar_count=10, last_entry_bar=0, entries_today=3)

    def test_over_limit_blocks_entry(self):
        rule = _make_entry_rule(max_entries_per_day=2)
        assert not can_accumulate(rule, [_make_lot()], bar_count=10, last_entry_bar=0, entries_today=5)

    def test_zero_entries_allows_first(self):
        rule = _make_entry_rule(max_entries_per_day=1)
        assert can_accumulate(rule, [_make_lot()], bar_count=10, last_entry_bar=0, entries_today=0)

    def test_single_mode_still_blocks(self):
        """max_entries_per_day doesn't override single mode."""
        rule = _make_entry_rule(max_entries_per_day=10, mode="single")
        assert not can_accumulate(rule, [_make_lot()], bar_count=10, last_entry_bar=0, entries_today=0)
