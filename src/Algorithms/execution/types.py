"""Typed data structures for the execution layer.

Replaces untyped dicts for lots, fills, equity points, and tracking state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FillInfo:
    """Fill data from LEAN's OnOrderEvent."""
    price: float
    quantity: float
    fee: float


@dataclass
class Lot:
    """An open position lot. Created on entry, closed on exit."""
    lot_id: int
    symbol: str
    direction: Literal["long", "short"]
    entry_time: str
    entry_price: float
    entry_bar: int
    quantity: float
    entry_fee: float = 0.0
    _exit_fee_share: float = 0.0


@dataclass
class ClosedLot:
    """A closed position lot with full PnL data."""
    lot_id: int
    symbol: str
    direction: Literal["long", "short"]
    entry_time: str
    entry_price: float
    entry_bar: int
    quantity: float
    entry_fee: float
    exit_time: str
    exit_price: float
    exit_bar: int
    pnl: float
    pnl_percent: float
    exit_fee: float
    total_fees: float
    exit_reason: str


@dataclass
class EquityPoint:
    """A single equity curve sample."""
    time: str
    equity: float
    cash: float
    holdings: float
    drawdown: float


@dataclass
class TradeStats:
    """Computed trade statistics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float


@dataclass
class TrackingState:
    """Mutable tracking state for the strategy runtime."""
    trades: list[ClosedLot] = field(default_factory=list)
    current_lots: list[Lot] = field(default_factory=list)
    last_entry_bar: int | None = None
    equity_curve: list[EquityPoint] = field(default_factory=list)
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    bar_count: int = 0
