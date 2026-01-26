# Phase 6: Typed Value Resolution — Implementation Plan

**Date**: 2026-01-26  
**Parent plan**: StrategyRuntime Full Typed Refactor (`~/.claude/plans/hazy-conjuring-rabbit.md`)  
**Scope**: LEAN only (vibe-trade-lean). No changes to execution or shared.

---

## 1. Goal

- Replace `StrategyRuntime._resolve_value(value_ref: dict, bar)` with a **typed** resolver that accepts **Pydantic ValueRef** and uses **only attribute access** (no `.get()`).
- Resolver lives in `indicators/resolvers.py`; exhaustive `match ref:` on the ValueRef union from vibe-trade-shared.
- Single boundary: at call site we have a dict (from IR JSON); we do `ValueRef.model_validate(value_ref)` once, then pass the typed ref into the resolver.
- Keep behavior identical so all existing E2E tests pass.
- Set up for Phase 7+ (EvalContext, extracting condition evaluators).

**Out of scope for Phase 6**: Parsing the full IR to StrategyIR in LEAN (Phase 11). We only parse value_ref dicts to ValueRef at the _resolve_value boundary.

---

## 2. Why No `.get()`: We Have Full Pydantic Models

vibe-trade-shared already defines the full ValueRef union and every variant (LiteralRef, PriceRef, IndicatorRef, etc.). So:

- **Resolver** receives `ref: ValueRef` and uses only typed attributes: `ref.value`, `ref.field`, `ref.indicator_id`, etc. No dict keys, no `.get()`.
- **Pattern matching** is on Pydantic types: `match ref: case LiteralRef(): return ref.value` — exhaustiveness is type-checkable.
- The **only** place we handle a dict is at the boundary: StrategyRuntime gets a dict from the IR; we call `ValueRef.model_validate(value_ref)` and pass the result into the resolver. That's one validation per resolution; no .get() inside the resolver at all.

---

## 3. Current State

| Location | Responsibility |
|----------|----------------|
| `StrategyRuntime._resolve_value(value_ref: dict, bar)` | ~120 lines; branches on `value_ref.get("type")` and `.get("field")`, etc. Uses `self.indicator_registry`, `self.state`, `self.Time`. |
| `conditions/registry.py` | Calls `runtime._resolve_value(condition.get("left"), bar)` (dict from IR). |
| `indicators/registry.py` | `resolve_indicator_value(category, indicator_or_data, field)` — keep using this for indicator/band/property. |

**ValueRef types (vibe-trade-shared)** — all have Pydantic models:

- LiteralRef, PriceRef, VolumeRef, TimeRef, StateRef  
- IndicatorRef, IndicatorBandRef, IndicatorPropertyRef  
- RollingWindowRef, IRExpression  

---

## 4. Target Design

- **New module**: `vibe-trade-lean/src/Algorithms/indicators/resolvers.py`
  - **Signature**:  
    `resolve_value(ref: ValueRef, bar, *, indicator_registry, state, current_time, rolling_windows=None) -> float`
  - **Implementation**: `match ref:` with cases for each Pydantic variant; use only `ref.<attr>`. For IndicatorRef/IndicatorBandRef/IndicatorPropertyRef, call existing `resolve_indicator_value(...)` from registry. For IRExpression, recursively call `resolve_value(ref.left, ...)` and `resolve_value(ref.right, ...)`.
  - **Imports**: `from vibe_trade_shared.models.ir import ValueRef, LiteralRef, PriceRef, ...` and `from .registry import resolve_indicator_value, IndicatorCategory`. Enums: `PriceField`, `IndicatorProperty` from shared.
- **StrategyRuntime._resolve_value(self, value_ref, bar)**:
  - Guard: `if not value_ref: return 0.0`
  - Parse once: `typed_ref = ValueRef.model_validate(value_ref)` (handles dict from IR).
  - Delegate: `return _resolve_value_impl(typed_ref, bar, indicator_registry=self.indicator_registry, state=self.state, current_time=self.Time, rolling_windows=self.rolling_windows)`
- **Container**: LEAN image must have access to **vibe-trade-shared** so we can import ValueRef and use Pydantic. See Task 6.5 (Docker/build).

---

## 5. Implementation Tasks

### Task 6.1: Add `indicators/resolvers.py` (typed; zero .get())

**File**: `vibe-trade-lean/src/Algorithms/indicators/resolvers.py`

- **Imports** (from shared and local):
  - `from vibe_trade_shared.models.ir import (... ValueRef, LiteralRef, PriceRef, VolumeRef, TimeRef, StateRef, IndicatorRef, IndicatorBandRef, IndicatorPropertyRef, RollingWindowRef, IRExpression, ...)` and `PriceField`, `IndicatorProperty` from enums.
  - `from .registry import resolve_indicator_value, IndicatorCategory`

- **Signature**:
  ```python
  def resolve_value(
      ref: ValueRef,
      bar,
      *,
      indicator_registry: dict,
      state: dict,
      current_time,
      rolling_windows: dict | None = None,
  ) -> float:
  ```

- **Behavior** (exhaustive `match ref:` — typed attribute access only):

  | Case | Action |
  |------|--------|
  | `LiteralRef()` | `return ref.value` |
  | `PriceRef()` | `ref.field` → bar.Open/High/Low/Close (use PriceField enum) |
  | `VolumeRef()` | `return float(bar.Volume)` |
  | `TimeRef()` | `ref.component` → current_time.hour / minute / weekday() |
  | `StateRef()` | `float(state.get(ref.state_id, 0))` (state is dict; only .get() here is on runtime state, not on ref) |
  | `IndicatorRef()` | lookup by `ref.indicator_id`; use `ref.field`; call `resolve_indicator_value(...)` |
  | `IndicatorBandRef()` | lookup by `ref.indicator_id`; return band by `ref.band` |
  | `IndicatorPropertyRef()` | lookup by `ref.indicator_id`; return property by `ref.property` |
  | `RollingWindowRef()` | from rolling_windows by `ref.indicator_id`, return window[ref.offset] or 0 |
  | `IRExpression()` | `left = resolve_value(ref.left, ...)`, `right = resolve_value(ref.right, ...)`, apply `ref.op` |

  If we need a fallback for unknown type (e.g. forward-compat), return 0.0 and optionally log; no .get() on `ref`.

### Task 6.2: Wire StrategyRuntime to resolvers

**File**: `vibe-trade-lean/src/Algorithms/StrategyRuntime.py`

- Add imports: `ValueRef` from vibe_trade_shared.models.ir; `from indicators import resolve_value as _resolve_value_impl`.
- Replace body of `_resolve_value(self, value_ref, bar)`:
  - `if not value_ref: return 0.0`
  - `typed_ref = ValueRef.model_validate(value_ref)`  # dict → Pydantic
  - `return _resolve_value_impl(typed_ref, bar, indicator_registry=self.indicator_registry, state=self.state, current_time=self.Time, rolling_windows=self.rolling_windows)`
- Delete the old ~120-line branch implementation.

### Task 6.3: Export from indicators package

**File**: `vibe-trade-lean/src/Algorithms/indicators/__init__.py`

- Export `resolve_value` from `.resolvers` (keep existing exports).

### Task 6.4: Run tests and sanity checks

- **Execution E2E**: `cd vibe-trade-execution && uv run pytest tests/e2e/ -v`
- **LEAN unit tests**: `cd vibe-trade-lean && uv run pytest tests/ -v`
- **.get() count**: `grep -c '\.get(' vibe-trade-lean/src/Algorithms/indicators/resolvers.py` — goal 0 on `ref` (only allowed: `state.get(ref.state_id, 0)` for runtime state dict).

### Task 6.5: Make vibe-trade-shared available in the LEAN container

The LEAN Dockerfile currently copies only Python files and does not install vibe-trade-shared, so the container has no `vibe_trade_shared` today. To use full Pydantic models we need shared in the image.

- **Option A (recommended)**  
  Build the service image from the **repo root** so we can COPY shared and install it:
  - Build context: repo root. Dockerfile path: `vibe-trade-lean/Dockerfile.service`.
  - In Dockerfile: `COPY vibe-trade-shared /vibe-trade-shared` then `RUN pip install /vibe-trade-shared` (or install from a stage). Ensure PYTHONPATH or install adds the package so `import vibe_trade_shared` works when LEAN runs the algorithm.
  - Root Makefile / CI may already build with context from root (e.g. `docker build -f vibe-trade-lean/Dockerfile.service .`); confirm and document.
- **Option B**  
  Vendor a minimal copy: copy only `value_refs.py` and required enums from shared into `vibe-trade-lean/src/Algorithms/ir_models/` and import from there in resolvers. Single source of truth stays in shared; LEAN gets a synced copy for the container build. No Dockerfile context change, but duplicate code to maintain.

Implement Option A (or B if build-from-root is not feasible) and document the choice in the plan.

---

## 6. Files Summary

| Action | File |
|--------|------|
| **Create** | `vibe-trade-lean/src/Algorithms/indicators/resolvers.py` (typed ValueRef; no .get() on ref) |
| **Modify** | `vibe-trade-lean/src/Algorithms/StrategyRuntime.py` (parse dict→ValueRef, delegate to resolver) |
| **Modify** | `vibe-trade-lean/src/Algorithms/indicators/__init__.py` (export resolve_value) |
| **Modify** | `vibe-trade-lean/Dockerfile.service` (and/or build context) so vibe-trade-shared is available in container |

---

## 7. Rollback / Safety

- No API or request/response changes; execution and shared are untouched.
- If regressions appear, revert the modified files and remove resolvers; call sites still use `runtime._resolve_value(ref, bar)` with the same dict input.

---

## 8. Success Criteria

- [ ] Resolver takes `ref: ValueRef` and uses only typed attribute access (no `.get()` on ref).
- [ ] All ValueRef variants (LiteralRef, PriceRef, VolumeRef, TimeRef, StateRef, IndicatorRef, IndicatorBandRef, IndicatorPropertyRef, RollingWindowRef, IRExpression) handled with exhaustive match.
- [ ] StrategyRuntime._resolve_value validates dict→ValueRef once and delegates to resolver.
- [ ] All execution E2E tests pass.
- [ ] LEAN unit tests pass.
- [ ] vibe-trade-shared is available in the LEAN container (Option A or B implemented).

---

## 9. Follow-ups (later phases)

- **Phase 7**: EvalContext; resolver invoked as `ctx.resolve_value(ref, bar)`.
- **Phase 11**: Parse full IR to StrategyIR in LEAN; condition evaluators then receive typed Condition instead of dict.

---

## 10. Reference

- ValueRef definitions: `vibe-trade-shared/src/vibe_trade_shared/models/ir/value_refs.py`
- Current implementation: `vibe-trade-lean/src/Algorithms/StrategyRuntime.py` lines ~1767–1878
- Indicator value resolution: `vibe-trade-lean/src/Algorithms/indicators/registry.py` `resolve_indicator_value`
- Architecture: `docs/ARCHITECTURE-AND-LOCAL-RUN.md` (in vibe-trade-execution)
