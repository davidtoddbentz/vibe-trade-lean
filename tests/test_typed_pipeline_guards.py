"""Guard tests: LEAN must consume typed models, not dicts.

These tests ensure the typed end-to-end pipeline is maintained:
- Indicator factories use attribute access (getattr), not dict .get()
- Condition evaluators don't use TypeAdapter or isinstance(dict) checks
- Indicator creation doesn't call model_dump()
"""

from pathlib import Path

ALGORITHMS_DIR = Path(__file__).resolve().parent.parent / "src" / "Algorithms"
INDICATORS_DIR = ALGORITHMS_DIR / "indicators"
CONDITIONS_DIR = ALGORITHMS_DIR / "conditions"


def test_no_dict_get_in_indicator_factories():
    """Indicator factories use getattr(), not ind_def.get()."""
    registry = (INDICATORS_DIR / "registry.py").read_text()
    assert "ind_def.get(" not in registry, (
        "Indicator factories must use getattr(ind_def, ...), not ind_def.get()"
    )


def test_no_dict_type_hint_in_indicator_factories():
    """Indicator factory signatures don't use dict[str, Any]."""
    registry = (INDICATORS_DIR / "registry.py").read_text()
    assert "ind_def: dict" not in registry, (
        "Indicator factories must accept typed models (Any), not dict[str, Any]"
    )


def test_no_model_dump_in_indicator_creation():
    """Indicator creation doesn't call model_dump() to convert typed models."""
    creation = (INDICATORS_DIR / "creation.py").read_text()
    # Check for actual .model_dump() calls, not mentions in comments
    import re
    calls = re.findall(r'\.model_dump\(', creation)
    assert not calls, (
        "creation.py must pass typed indicator models directly, not model_dump()"
    )


def test_no_type_adapter_in_condition_registry():
    """Condition registry doesn't use TypeAdapter for re-parsing typed conditions."""
    registry = (CONDITIONS_DIR / "registry.py").read_text()
    assert "TypeAdapter" not in registry, (
        "Condition evaluators receive typed conditions - no TypeAdapter needed"
    )


def test_no_isinstance_dict_in_condition_registry():
    """Condition evaluators don't check isinstance(condition, dict)."""
    registry = (CONDITIONS_DIR / "registry.py").read_text()
    assert "isinstance(condition, dict)" not in registry, (
        "All conditions are typed from model_validate() - no dict checks needed"
    )
