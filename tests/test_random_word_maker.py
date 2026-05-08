"""Tests for utilz.random_word_maker tiered name-selection."""
import importlib.util
import sys
import yaml
import pytest
from pathlib import Path

# Import random_word_maker directly to avoid triggering utilz/__init__.py,
# which depends on optional heavy packages (torch, etc.) not available in CI.
_rwm_path = Path(__file__).parent.parent / "utilz" / "random_word_maker.py"
_spec = importlib.util.spec_from_file_location("random_word_maker", _rwm_path)
rwm = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("random_word_maker", rwm)
_spec.loader.exec_module(rwm)


# ---------------------------------------------------------------------------
# random_fake_word
# ---------------------------------------------------------------------------

def test_fake_word_length():
    for _ in range(50):
        w = rwm.random_fake_word(3, 6)
        assert 3 <= len(w) <= 6


def test_fake_word_fixed_length():
    for _ in range(20):
        w = rwm.random_fake_word(4, 4)
        assert len(w) == 4


# ---------------------------------------------------------------------------
# YAML structure
# ---------------------------------------------------------------------------

def test_yaml_files_exist():
    assert rwm.SHORT_NAMES_YAML.exists(), f"{rwm.SHORT_NAMES_YAML} not found"
    assert rwm.LONG_NAMES_YAML.exists(), f"{rwm.LONG_NAMES_YAML} not found"


def test_yaml_has_common_and_uncommon():
    for path in (rwm.SHORT_NAMES_YAML, rwm.LONG_NAMES_YAML):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "common" in data, f"{path.name} missing 'common' key"
        assert "uncommon" in data, f"{path.name} missing 'uncommon' key"
        assert len(data["common"]) == 1000, (
            f"{path.name} 'common' should have 1000 entries, got {len(data['common'])}"
        )
        assert len(data["uncommon"]) > 0, f"{path.name} 'uncommon' is empty"


# ---------------------------------------------------------------------------
# _collect_words
# ---------------------------------------------------------------------------

def test_collect_common_words_length_filter():
    rwm._yaml_cache.clear()
    words = rwm._collect_words(3, 5, "common")
    assert words, "common bucket should return words for length 3-5"
    for w in words:
        assert 3 <= len(w) <= 5, f"Word '{w}' outside expected range"


def test_collect_uncommon_words_long():
    rwm._yaml_cache.clear()
    words = rwm._collect_words(6, 10, "uncommon")
    assert words, "uncommon bucket should return words for length 6-10"
    for w in words:
        assert 6 <= len(w) <= 10


# ---------------------------------------------------------------------------
# random_real_word - basic behaviour
# ---------------------------------------------------------------------------

def test_random_real_word_returns_string():
    rwm._yaml_cache.clear()
    w = rwm.random_real_word(3, 5)
    assert isinstance(w, str)
    assert len(w) >= 3


def test_random_real_word_respects_length():
    rwm._yaml_cache.clear()
    for _ in range(20):
        w = rwm.random_real_word(3, 5)
        assert 3 <= len(w) <= 5


# ---------------------------------------------------------------------------
# Tiered selection: common before uncommon
# ---------------------------------------------------------------------------

def _patch_paths(tmp_path, short_data, long_data):
    """Write fake YAML files and return (short_path, long_path)."""
    short_path = tmp_path / "names_short.yaml"
    long_path = tmp_path / "names_long.yaml"
    short_path.write_text(yaml.dump(short_data), encoding="utf-8")
    long_path.write_text(yaml.dump(long_data), encoding="utf-8")
    return short_path, long_path


def test_common_exhaustion_falls_back_to_uncommon(tmp_path, monkeypatch):
    """When all common names are in used_names, pick from uncommon."""
    short_path, long_path = _patch_paths(
        tmp_path,
        {"common": ["Beta"], "uncommon": ["Zyx"]},
        {"common": [], "uncommon": []},
    )
    monkeypatch.setattr(rwm, "SHORT_NAMES_YAML", short_path)
    monkeypatch.setattr(rwm, "LONG_NAMES_YAML", long_path)
    rwm._yaml_cache.clear()

    result = rwm.random_real_word(3, 5, caller_context="proj", used_names={"Beta"})
    assert result == "Zyx"


def test_returns_common_when_not_used(tmp_path, monkeypatch):
    """With no used_names, common bucket is preferred."""
    short_path, long_path = _patch_paths(
        tmp_path,
        {"common": ["Kalo"], "uncommon": ["Zyx"]},
        {"common": [], "uncommon": []},
    )
    monkeypatch.setattr(rwm, "SHORT_NAMES_YAML", short_path)
    monkeypatch.setattr(rwm, "LONG_NAMES_YAML", long_path)
    rwm._yaml_cache.clear()

    result = rwm.random_real_word(4, 4)
    assert result == "Kalo"


def test_fake_word_fallback_when_all_used(tmp_path, monkeypatch):
    """When both buckets are exhausted, a fake word is generated."""
    short_path, long_path = _patch_paths(
        tmp_path,
        {"common": ["Beta"], "uncommon": ["Zyx"]},
        {"common": [], "uncommon": []},
    )
    monkeypatch.setattr(rwm, "SHORT_NAMES_YAML", short_path)
    monkeypatch.setattr(rwm, "LONG_NAMES_YAML", long_path)
    rwm._yaml_cache.clear()

    result = rwm.random_real_word(3, 5, used_names={"Beta", "Zyx"})
    assert isinstance(result, str)
    assert len(result) >= 3


def test_caller_context_ignored_without_used_names(tmp_path, monkeypatch):
    """caller_context alone has no effect; used_names drives exclusion."""
    short_path, long_path = _patch_paths(
        tmp_path,
        {"common": ["Kalo"], "uncommon": ["Zyx"]},
        {"common": [], "uncommon": []},
    )
    monkeypatch.setattr(rwm, "SHORT_NAMES_YAML", short_path)
    monkeypatch.setattr(rwm, "LONG_NAMES_YAML", long_path)
    rwm._yaml_cache.clear()

    result = rwm.random_real_word(4, 4, caller_context="my-project")
    assert result == "Kalo"
