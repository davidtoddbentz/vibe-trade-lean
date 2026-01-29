"""Trading cost utilities.

Fees and slippage are now handled by LEAN's native FeeModel and SlippageModel,
wired up in StrategyRuntime.Initialize().

- PercentageFeeModel (defined in StrategyRuntime.py) -> security.SetFeeModel()
- ConstantSlippageModel (LEAN built-in) -> security.SetSlippageModel()
"""
