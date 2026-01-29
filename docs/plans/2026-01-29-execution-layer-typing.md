# Execution Layer Typing & ExecutionContext Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace all `Any` types and untyped dicts in the execution layer with proper dataclasses, typed IR imports, and an `ExecutionContext` that eliminates 13-parameter function signatures.

**Architecture:** Introduce typed dataclasses (`Lot`, `ClosedLot`, `FillInfo`, `TrackingState`, `EquityPoint`) in a new `execution/types.py` module. Create an `ExecutionContext` dataclass in `execution/context.py` that bundles LEAN primitives (portfolio, symbol, SetHoldings, MarketOrder, Liquidate, Log, get_last_fill). Update all function signatures to use real IR types from `vibe_trade_shared.models.ir` and the new typed structures. Use `TYPE_CHECKING` imports for LEAN types (already established pattern in `indicators/registry.py`).

**Tech Stack:** Python 3.10+, dataclasses, `TYPE_CHECKING` for LEAN types, `vibe_trade_shared.models.ir` for IR types

**Constraints:**
- LEAN types (`Symbol`, `Portfolio`, `TradeBar`, `Resolution`) are only available at runtime inside Docker — use `TYPE_CHECKING` guard for type hints
- All 88 E2E tests must pass after each task (run from `vibe-trade-execution/`)
- The `generate_report` output is JSON — final report must still be `dict` for serialization, but internal structures use dataclasses
- No behavioral changes — pure refactoring

---

### Task 1: Create typed data structures (`execution/types.py`)

**Files:**
- Create: `vibe-trade-lean/src/Algorithms/execution/types.py`

**Step 1: Create the types module**

```python
"""Typed data structures for the execution layer.

Replaces untyped dicts for lots, fills, equity points, and tracking state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime


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
    _exit_fee_share: float = 0.0  # Set during exit fee distribution


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
```

**Step 2: Commit**

```bash
git add vibe-trade-lean/src/Algorithms/execution/types.py
git commit -m "feat: add typed data structures for execution layer

Introduces Lot, ClosedLot, FillInfo, EquityPoint, TradeStats,
and TrackingState dataclasses to replace untyped dicts."
```

---

### Task 2: Create `ExecutionContext` (`execution/context.py`)

**Files:**
- Create: `vibe-trade-lean/src/Algorithms/execution/context.py`

**Step 1: Create the context module**

The `ExecutionContext` bundles LEAN primitives that are currently passed as 8+ separate `Any` parameters to every function. LEAN types use `TYPE_CHECKING` guard (pattern from `indicators/registry.py`).

```python
"""ExecutionContext — bundles LEAN primitives for the execution layer.

Eliminates passing 8+ separate Any parameters to every function.
LEAN types are imported under TYPE_CHECKING (runtime-only in Docker).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from execution.types import FillInfo

if TYPE_CHECKING:
    from QuantConnect import Symbol
    from QuantConnect.Securities import SecurityManager
    from QuantConnect.Securities.SecurityPortfolioManager import SecurityPortfolioManager


@dataclass
class ExecutionContext:
    """LEAN execution primitives bundled for clean function signatures.

    Created once in StrategyRuntime.Initialize(), passed to execution functions.
    """

    symbol: Any  # Symbol at runtime, Any for non-LEAN contexts
    portfolio: Any  # SecurityPortfolioManager at runtime
    securities: Any  # SecurityManager at runtime
    set_holdings: Callable  # self.SetHoldings
    market_order: Callable  # self.MarketOrder
    liquidate: Callable  # self.Liquidate
    log: Callable[[str], None]  # self.Log
    get_last_fill: Callable[[], FillInfo | None]  # self._get_and_clear_last_fill
```

Note: We use `Any` for `symbol`, `portfolio`, `securities` because these are LEAN runtime objects that can't be instantiated outside Docker. The `Callable` types provide the real contract. If we later add type stubs for LEAN, we can upgrade these to proper types.

**Step 2: Update `execution/__init__.py` to export the new types**

Read the current `execution/__init__.py` and add exports for `ExecutionContext`, `FillInfo`, `Lot`, `ClosedLot`, etc.

**Step 3: Commit**

```bash
git add vibe-trade-lean/src/Algorithms/execution/context.py vibe-trade-lean/src/Algorithms/execution/__init__.py
git commit -m "feat: add ExecutionContext to bundle LEAN primitives

Replaces 8+ separate Any params with a single typed context object."
```

---

### Task 3: Migrate `trades/tracking.py` to typed `Lot` and `ClosedLot`

**Files:**
- Modify: `vibe-trade-lean/src/Algorithms/trades/tracking.py`

This is the core data model change. `create_lot` returns `Lot`, `close_lots` takes `list[Lot]` and returns `list[ClosedLot]`, `calculate_trade_stats` takes `list[ClosedLot]`.

**Step 1: Update `create_lot`**

Change return type from `dict[str, Any]` to `Lot`:

```python
from execution.types import Lot, ClosedLot, TradeStats

def create_lot(
    lot_id: int,
    symbol: str,
    direction: str,
    entry_time: Any,
    entry_price: float,
    entry_bar: int,
    quantity: float,
    entry_fee: float,
) -> Lot:
    return Lot(
        lot_id=lot_id,
        symbol=symbol,
        direction=direction,
        entry_time=str(entry_time),
        entry_price=entry_price,
        entry_bar=entry_bar,
        quantity=quantity,
        entry_fee=entry_fee,
    )
```

**Step 2: Update `close_lots`**

Change to accept `list[Lot]` and return `list[ClosedLot]`:

```python
def close_lots(
    lots: list[Lot],
    exit_price: float,
    exit_time: Any,
    exit_bar: int,
    exit_reason: str,
) -> list[ClosedLot]:
    closed = []
    for lot in lots:
        final_exit_price = exit_price
        exit_fee = lot._exit_fee_share
        total_fees = lot.entry_fee + exit_fee

        if lot.direction == "short":
            gross_pnl = (lot.entry_price - final_exit_price) * lot.quantity
        else:
            gross_pnl = (final_exit_price - lot.entry_price) * lot.quantity

        pnl = gross_pnl - total_fees
        entry_value = lot.entry_price * lot.quantity
        pnl_pct = (pnl / entry_value * 100) if entry_value > 0 else 0.0

        closed.append(ClosedLot(
            lot_id=lot.lot_id,
            symbol=lot.symbol,
            direction=lot.direction,
            entry_time=lot.entry_time,
            entry_price=lot.entry_price,
            entry_bar=lot.entry_bar,
            quantity=lot.quantity,
            entry_fee=lot.entry_fee,
            exit_time=str(exit_time),
            exit_price=final_exit_price,
            exit_bar=exit_bar,
            pnl=pnl,
            pnl_percent=pnl_pct,
            exit_fee=exit_fee,
            total_fees=total_fees,
            exit_reason=exit_reason,
        ))
    return closed
```

**Step 3: Update `calculate_trade_stats`**

Change to accept `list[ClosedLot]` and return `TradeStats`:

```python
def calculate_trade_stats(trades: list[ClosedLot]) -> TradeStats:
    if not trades:
        return TradeStats(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, avg_win_pct=0.0, avg_loss_pct=0.0, profit_factor=0.0,
        )

    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]
    win_rate = len(winning) / len(trades) * 100

    avg_win = (sum(t.pnl_percent for t in winning) / len(winning)) if winning else 0.0
    avg_loss = (sum(t.pnl_percent for t in losing) / len(losing)) if losing else 0.0

    total_wins = sum(t.pnl for t in winning)
    total_losses = abs(sum(t.pnl for t in losing))
    profit_factor = (total_wins / total_losses) if total_losses > 0 else float("inf")

    return TradeStats(
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=profit_factor,
    )
```

**Step 4: Update `generate_report`**

Change trades param to `list[ClosedLot]`, equity_curve to `list[EquityPoint]`. The output dict stays as `dict` for JSON serialization, but converts from typed structures:

```python
from execution.types import ClosedLot, EquityPoint, TradeStats
from dataclasses import asdict

def generate_report(
    trades: list[ClosedLot],
    equity_curve: list[EquityPoint],
    initial_cash: float,
    final_equity: float,
    max_drawdown: float,
    strategy_id: str,
    strategy_name: str,
    symbol: str,
    log_func: Callable[[str], None],
) -> dict:
    # ... compute stats using TradeStats ...
    stats = calculate_trade_stats(trades)

    # Convert to dicts for JSON serialization
    output = {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "symbol": symbol,
        "initial_cash": initial_cash,
        "final_equity": float(final_equity),
        "total_return_pct": total_return,
        "max_drawdown_pct": max_drawdown,
        "statistics": asdict(stats),
        "trades": [asdict(t) for t in trades],
        "equity_curve": [asdict(e) for e in equity_curve],
    }
    # ... accumulation summary logic unchanged ...
    return output
```

**Step 5: Update `close_lots_at_end`**

```python
def close_lots_at_end(
    lots: list[Lot],
    exit_price: float,
    exit_time: Any,
    exit_bar: int,
    log_func: Callable[[str], None],
) -> list[ClosedLot]:
    return close_lots(
        lots=lots,
        exit_price=exit_price,
        exit_time=exit_time,
        exit_bar=exit_bar,
        exit_reason="end_of_backtest",
    )
```

**Step 6: Build LEAN image and run E2E tests**

```bash
cd /home/david/workspace/vibe-trade && make lean-build-workspace
# Start 3 containers on ports 8081-8083
cd vibe-trade-execution && uv run pytest tests/test_e2e_prototype.py -n 3 -v
```

Expected: 88/88 pass

**Step 7: Commit**

```bash
git add vibe-trade-lean/src/Algorithms/trades/tracking.py
git commit -m "refactor: migrate tracking.py to typed Lot and ClosedLot dataclasses

Replaces dict[str, Any] with Lot (open), ClosedLot (closed),
TradeStats, and EquityPoint dataclasses. Attribute access
instead of string key lookups."
```

---

### Task 4: Migrate `execution/orchestration.py` to typed signatures + ExecutionContext

**Files:**
- Modify: `vibe-trade-lean/src/Algorithms/execution/orchestration.py`

This is the biggest change. `execute_entry` goes from 13 params to ~6, `execute_exit` from 11 to ~5.

**Step 1: Update imports and `execute_entry` signature**

```python
from __future__ import annotations

from typing import Any, Callable

from vibe_trade_shared.models.ir import (
    EntryRule,
    ExitRule,
    OverlayRule,
    SetHoldingsAction,
    MarketOrderAction,
    Condition,
    StateOp,
)
from execution.types import Lot, ClosedLot, FillInfo
from execution.context import ExecutionContext
from trades import create_lot
from position import apply_scale_in, apply_overlay_scale, compute_overlay_scale


def execute_entry(
    entry_rule: EntryRule | None,
    evaluate_condition: Callable[[Condition, Any], bool],
    bar: Any,  # TradeBar at runtime
    current_lots: list[Lot],
    bar_count: int,
    ctx: ExecutionContext,
    current_time: Any,
    execute_action_func: Callable,
    execute_state_op: Callable[[StateOp, Any], None],
    overlays: list[OverlayRule],
) -> tuple[list[Lot], int | None]:
```

Key changes:
- `entry_rule: Any` → `EntryRule | None`
- `current_lots: list[Any]` → `list[Lot]`
- `overlays: list[Any]` → `list[OverlayRule]`
- `symbol`, `portfolio`, `securities`, `set_holdings_func`, `market_order_func`, `liquidate_func`, `log_func`, `get_last_fill` all collapse into `ctx: ExecutionContext`
- `evaluate_condition_func: Any` → `Callable[[Condition, Any], bool]`
- `execute_state_op_func: Any` → `Callable[[StateOp, Any], None]`

**Step 2: Update `execute_entry` body**

Replace individual params with `ctx.*`:
- `portfolio[symbol]` → `ctx.portfolio[ctx.symbol]`
- `log_func(...)` → `ctx.log(...)`
- `get_last_fill()` → `ctx.get_last_fill()`

Replace `getattr()` with typed access:
- `getattr(entry_rule, "on_fill", None)` → `entry_rule.on_fill`

Replace fill dict access with `FillInfo` attributes:
- `fill["price"]` → `fill.price`
- `fill["fee"]` → `fill.fee`

**Step 3: Update `execute_exit` signature**

```python
def execute_exit(
    exit_rules: list[ExitRule],
    evaluate_condition: Callable[[Condition, Any], bool],
    bar: Any,  # TradeBar at runtime
    current_lots: list[Lot],
    bar_count: int,
    ctx: ExecutionContext,
    current_time: Any,
    close_lots_func: Callable[..., list[ClosedLot]],
) -> tuple[list[Lot], list[ClosedLot]]:
```

Key changes:
- `exit_rules: list[Any]` → `list[ExitRule]`
- `execute_action_func`, `log_func`, `get_last_fill` collapse into `ctx`
- Return type `list[Any]` → `list[ClosedLot]`

**Step 4: Update `execute_exit` body**

Replace `getattr()` with typed access:
- `getattr(x, "priority", 0)` → `x.priority or 0` (ExitRule has `priority: int`)
- `getattr(exit_rule, "id", "unknown")` → `exit_rule.id`

Replace lot dict mutation with attribute assignment:
- `lot["_exit_fee_share"] = ...` → `lot._exit_fee_share = ...`

Replace fill dict access with `FillInfo` attributes.

**Step 5: Build LEAN image and run E2E tests**

```bash
cd /home/david/workspace/vibe-trade && make lean-build-workspace
cd vibe-trade-execution && uv run pytest tests/test_e2e_prototype.py -n 3 -v
```

Expected: 88/88 pass

**Step 6: Commit**

```bash
git add vibe-trade-lean/src/Algorithms/execution/orchestration.py
git commit -m "refactor: migrate orchestration.py to ExecutionContext and typed IR

execute_entry: 13 params -> 10 (ctx bundles LEAN primitives)
execute_exit: 11 params -> 8
All params now typed with IR models and dataclasses."
```

---

### Task 5: Update `StrategyRuntime.py` to use `ExecutionContext` and typed structures

**Files:**
- Modify: `vibe-trade-lean/src/Algorithms/StrategyRuntime.py`

**Step 1: Create `ExecutionContext` in `Initialize()`**

After setting up symbols and indicators, create the context:

```python
from execution.context import ExecutionContext
from execution.types import FillInfo, TrackingState

# In Initialize(), after symbol setup:
self._exec_ctx = ExecutionContext(
    symbol=self.symbol,
    portfolio=self.Portfolio,
    securities=self.Securities,
    set_holdings=self.SetHoldings,
    market_order=self.MarketOrder,
    liquidate=self.Liquidate,
    log=self.Log,
    get_last_fill=self._get_and_clear_last_fill,
)
```

**Step 2: Update `_get_and_clear_last_fill` to return `FillInfo | None`**

```python
def _get_and_clear_last_fill(self) -> FillInfo | None:
    fill = self._last_fill
    self._last_fill = None
    return fill
```

**Step 3: Update `OnOrderEvent` to store `FillInfo`**

```python
def OnOrderEvent(self, orderEvent):
    if orderEvent.Status != OrderStatus.Filled:
        return
    fill_price = float(orderEvent.FillPrice)
    fill_qty = float(orderEvent.FillQuantity)
    order_fee = 0.0
    try:
        order_fee = float(orderEvent.OrderFee.Value.Amount)
    except (AttributeError, Exception):
        pass
    self._last_fill = FillInfo(
        price=fill_price,
        quantity=fill_qty,
        fee=order_fee,
    )
```

**Step 4: Update `_evaluate_entry` and `_evaluate_exits` call sites**

Replace the 13-param calls with the new signatures:

```python
def _evaluate_entry(self, bar):
    self.current_lots, new_entry_bar = execute_entry(
        entry_rule=self.entry_rule,
        evaluate_condition=self._evaluate_condition,
        bar=bar,
        current_lots=self.current_lots,
        bar_count=self.bar_count,
        ctx=self._exec_ctx,
        current_time=self.Time,
        execute_action_func=lambda action, b=None: self._execute_action(action, b),
        execute_state_op=self._execute_state_op,
        overlays=self.overlays,
    )
    if new_entry_bar is not None:
        self.last_entry_bar = new_entry_bar
```

```python
def _evaluate_exits(self, bar):
    self.current_lots, closed_lots_list = execute_exit(
        exit_rules=self.exit_rules,
        evaluate_condition=self._evaluate_condition,
        bar=bar,
        current_lots=self.current_lots,
        bar_count=self.bar_count,
        ctx=self._exec_ctx,
        current_time=self.Time,
        close_lots_func=close_lots,
    )
    self.trades.extend(closed_lots_list)
```

Note: The lambda wrapper for `execute_action_func` stays for now — it wraps `self._execute_action` which adds `bar` context. This can be further cleaned up in the orchestration restructure (Option C), but it's not a typing issue.

**Step 5: Update tracking state initialization**

Replace dict unpacking with `TrackingState`:

```python
tracking = TrackingState(peak_equity=initial_cash)
self.trades = tracking.trades
self.current_lots = tracking.current_lots
self.last_entry_bar = tracking.last_entry_bar
self.equity_curve = tracking.equity_curve
self.peak_equity = tracking.peak_equity
self.max_drawdown = tracking.max_drawdown
self.bar_count = tracking.bar_count
```

Remove `setup_tracking()` function from `initialization/setup.py` since `TrackingState()` replaces it. Update `initialization/__init__.py` to remove the export.

**Step 6: Build LEAN image and run E2E tests**

```bash
cd /home/david/workspace/vibe-trade && make lean-build-workspace
cd vibe-trade-execution && uv run pytest tests/test_e2e_prototype.py -n 3 -v
```

Expected: 88/88 pass

**Step 7: Commit**

```bash
git add vibe-trade-lean/src/Algorithms/StrategyRuntime.py vibe-trade-lean/src/Algorithms/initialization/setup.py vibe-trade-lean/src/Algorithms/initialization/__init__.py
git commit -m "refactor: StrategyRuntime uses ExecutionContext and typed TrackingState

Replaces dict-based tracking with TrackingState dataclass.
OnOrderEvent stores FillInfo instead of dict.
Entry/exit calls use ExecutionContext instead of 13 separate params."
```

---

### Task 6: Type the remaining function signatures

**Files:**
- Modify: `vibe-trade-lean/src/Algorithms/execution/actions.py`
- Modify: `vibe-trade-lean/src/Algorithms/position/sizing.py`
- Modify: `vibe-trade-lean/src/Algorithms/position/equity.py`
- Modify: `vibe-trade-lean/src/Algorithms/gates/evaluation.py`
- Modify: `vibe-trade-lean/src/Algorithms/state/operations.py`
- Modify: `vibe-trade-lean/src/Algorithms/initialization/setup.py`

**Step 1: Update `execution/actions.py`**

Replace individual LEAN params with `ExecutionContext`:

```python
from execution.context import ExecutionContext

def execute_action(
    action: EntryAction | ExitAction,
    ctx: ExecutionContext,
    bar: Any = None,
) -> None:
```

Body changes:
- `set_holdings_func(symbol, ...)` → `ctx.set_holdings(ctx.symbol, ...)`
- `market_order_func(symbol, ...)` → `ctx.market_order(ctx.symbol, ...)`
- `liquidate_func(symbol)` → `ctx.liquidate(ctx.symbol)`
- `log_func(...)` → `ctx.log(...)`
- `securities[symbol].Price` → `ctx.securities[ctx.symbol].Price`

Then update `StrategyRuntime._execute_action` to pass `ctx`:

```python
def _execute_action(self, action, bar=None):
    execute_action(action=action, ctx=self._exec_ctx, bar=bar)
```

**Step 2: Update `position/sizing.py`**

Replace `Any` with IR types:

```python
from vibe_trade_shared.models.ir import EntryRule, OverlayRule, Condition
from execution.types import Lot

def can_accumulate(
    entry_rule: EntryRule | None,
    current_lots: list[Lot],
    bar_count: int,
    last_entry_bar: int | None,
) -> bool:

def apply_scale_in(
    action: SetHoldingsAction,
    current_lots: list[Lot],
    log_func: Callable[[str], None],
) -> SetHoldingsAction:

def compute_overlay_scale(
    overlays: list[OverlayRule],
    evaluate_condition_func: Callable[[Condition, Any], bool],
    bar: Any,
    log_func: Callable[[str], None],
) -> float:
```

Replace `getattr()` with typed access:
- `getattr(overlay, "target_roles", ...)` → `overlay.target_roles or ["entry", "exit"]`
- `getattr(overlay, "scale_size_frac", 1.0)` → `overlay.scale_size_frac or 1.0`
- `getattr(overlay, "id", "unknown")` → `overlay.id or "unknown"`

**Step 3: Update `position/equity.py`**

```python
from execution.types import EquityPoint

def track_equity(
    equity: float,
    cash: float,
    holdings: float,
    drawdown: float,
    current_time: Any,
    bar_count: int,
    resolution: Any,  # Resolution at runtime
    equity_curve: list[EquityPoint],
    peak_equity: float,
    max_drawdown: float,
) -> tuple[float, float]:
```

Replace dict append with `EquityPoint`:
- `equity_curve.append({"time": ..., "equity": ...})` → `equity_curve.append(EquityPoint(time=..., equity=...))`

**Step 4: Update `gates/evaluation.py`**

```python
from vibe_trade_shared.models.ir import GateRule, Condition

def evaluate_gates(
    gates: list[GateRule],
    evaluate_condition_func: Callable[[Condition, Any], bool],
    bar: Any,
) -> bool:
```

Replace `getattr()`:
- `getattr(gate, "mode", "allow")` → `gate.mode or "allow"`

**Step 5: Update `state/operations.py`**

Already has `StateOp` typed. Just update callbacks:

```python
def execute_state_op(
    op: StateOp,
    bar: Any,
    state: dict[str, float],
    resolve_value_func: Callable,
    evaluate_condition_func: Callable,
    log_func: Callable[[str], None],
) -> None:
```

Change `state: dict[str, Any]` → `dict[str, float]` since state values are always floats.

**Step 6: Update `initialization/setup.py`**

Replace `Any` with IR types and typed returns:

```python
from vibe_trade_shared.models.ir import StrategyIR, EntryRule, ExitRule, GateRule, OverlayRule, StateOp

def setup_symbols(
    ir: StrategyIR,
    add_symbol_func: Callable,
    normalize_symbol_func: Callable[[str], str],
    log_func: Callable[[str], None],
) -> tuple[Any, dict[str, Any]]:  # Symbol + symbols dict

def setup_rules(ir: StrategyIR) -> dict:
    return {
        "entry_rule": ir.entry,
        "exit_rules": ir.exits or [],
        "gates": ir.gates or [],
        "overlays": ir.overlays or [],
        "on_bar_ops": ir.on_bar or [],
        "on_bar_invested_ops": ir.on_bar_invested or [],
    }

def setup_trading_costs(
    ir: StrategyIR,
    log_func: Callable[[str], None],
) -> dict[str, float]:
```

Replace all `getattr(ir, "field", default)` with `ir.field or default` since `ir` is now typed `StrategyIR`.

**Step 7: Build LEAN image and run E2E tests**

```bash
cd /home/david/workspace/vibe-trade && make lean-build-workspace
cd vibe-trade-execution && uv run pytest tests/test_e2e_prototype.py -n 3 -v
```

Expected: 88/88 pass

**Step 8: Commit**

```bash
git add vibe-trade-lean/src/Algorithms/execution/actions.py vibe-trade-lean/src/Algorithms/position/sizing.py vibe-trade-lean/src/Algorithms/position/equity.py vibe-trade-lean/src/Algorithms/gates/evaluation.py vibe-trade-lean/src/Algorithms/state/operations.py vibe-trade-lean/src/Algorithms/initialization/setup.py
git commit -m "refactor: type all remaining function signatures

Replace Any with IR types (EntryRule, ExitRule, GateRule, etc.),
Callable types for callbacks, Lot/ClosedLot for lot lists.
Replace getattr() with typed attribute access throughout."
```

---

### Task 7: Update `conditions/context.py` EvalContext typing

**Files:**
- Modify: `vibe-trade-lean/src/Algorithms/conditions/context.py`

**Step 1: Improve EvalContext types**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from vibe_trade_shared.models.ir import Condition


@dataclass
class EvalContext:
    """Minimal context needed by condition evaluators."""

    resolve_value: Callable[[Any, Any], float]  # (ValueRef, bar) -> float
    evaluate_condition: Callable[[Condition, Any], bool]  # (Condition, bar) -> bool
    state: dict[str, float]
    current_time: Any  # datetime at runtime
    cross_state: dict[str, tuple[float, float]]
    rolling_windows: dict[str, Any]  # LEAN RollingWindow objects
    indicators: dict[str, Any]  # LEAN indicator objects
    breakout_prev_max: dict[str, float]
    breakout_prev_min: dict[str, float]
```

Changes:
- `state: dict[str, Any]` → `dict[str, float]`
- `evaluate_condition` callback gets `Condition` type

Note: `rolling_windows` and `indicators` stay `Any` because they hold LEAN runtime objects with no Python type stubs.

**Step 2: Build LEAN image and run E2E tests**

```bash
cd /home/david/workspace/vibe-trade && make lean-build-workspace
cd vibe-trade-execution && uv run pytest tests/test_e2e_prototype.py -n 3 -v
```

Expected: 88/88 pass

**Step 3: Commit**

```bash
git add vibe-trade-lean/src/Algorithms/conditions/context.py
git commit -m "refactor: tighten EvalContext typing

state dict values typed as float, evaluate_condition takes Condition."
```

---

### Task 8: Final verification and cleanup

**Files:**
- Verify: all modified files

**Step 1: Search for remaining `Any` in execution layer**

```bash
grep -rn ": Any" vibe-trade-lean/src/Algorithms/execution/ vibe-trade-lean/src/Algorithms/trades/ vibe-trade-lean/src/Algorithms/position/ vibe-trade-lean/src/Algorithms/gates/ vibe-trade-lean/src/Algorithms/state/ vibe-trade-lean/src/Algorithms/initialization/
```

Verify that remaining `Any` usages are justified (LEAN runtime types, `bar` parameter).

**Step 2: Search for remaining `dict[str, Any]` returns**

```bash
grep -rn "dict\[str, Any\]" vibe-trade-lean/src/Algorithms/
```

Only `generate_report` should return `dict` (for JSON serialization). Everything internal should use dataclasses.

**Step 3: Search for remaining `getattr()`**

```bash
grep -rn "getattr(" vibe-trade-lean/src/Algorithms/execution/ vibe-trade-lean/src/Algorithms/trades/ vibe-trade-lean/src/Algorithms/position/ vibe-trade-lean/src/Algorithms/gates/ vibe-trade-lean/src/Algorithms/initialization/
```

Should be zero (all replaced with typed attribute access).

**Step 4: Full E2E test run**

```bash
cd /home/david/workspace/vibe-trade && make lean-build-workspace
cd vibe-trade-execution && uv run pytest tests/test_e2e_prototype.py -n 3 -v
```

Expected: 88/88 pass

**Step 5: Commit any remaining cleanup**

```bash
git add -A
git commit -m "refactor: final cleanup after execution layer typing migration"
```
