"""EmbeddingCache: loads embeddings, cosine search, thresholds, multi-embedding."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.recognition.embedding_cache import EmbeddingCache  # noqa: E402


class _FakeDB:
    """Duck-typed stand-in for FaceDatabase (no SQLite needed)."""

    def __init__(self, rows: list[tuple[str, np.ndarray]]) -> None:
        self._rows = rows

    def get_all_embeddings(self) -> list[tuple[str, np.ndarray]]:
        return list(self._rows)


def _vec(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def test_loads_embeddings_and_counts():
    db = _FakeDB([("CS1", _vec(1, 0, 0)), ("CS2", _vec(0, 1, 0))])
    cache = EmbeddingCache.load(db)
    assert len(cache) == 2
    assert cache.unique_students() == 2


def test_find_best_returns_correct_student():
    db = _FakeDB([("CS1", _vec(1, 0, 0)), ("CS2", _vec(0, 1, 0))])
    cache = EmbeddingCache.load(db)
    match = cache.find_best(_vec(0.9, 0.1, 0.0), threshold=0.45)
    assert match is not None
    code, score = match
    assert code == "CS1"
    assert score > 0.9


def test_find_best_below_threshold_returns_none():
    db = _FakeDB([("CS1", _vec(1, 0, 0))])
    cache = EmbeddingCache.load(db)
    # Orthogonal query -> cosine ~0 -> below threshold.
    assert cache.find_best(_vec(0, 1, 0), threshold=0.45) is None


def test_empty_cache_returns_none():
    cache = EmbeddingCache.load(_FakeDB([]))
    assert len(cache) == 0
    assert cache.find_best(_vec(1, 0, 0)) is None


def test_degenerate_embeddings_are_skipped():
    db = _FakeDB([("CS1", _vec(0, 0, 0)), ("CS2", _vec(0, 0, 1))])
    cache = EmbeddingCache.load(db)
    assert len(cache) == 1  # zero-norm row dropped
    match = cache.find_best(_vec(0, 0, 2), threshold=0.45)
    assert match is not None and match[0] == "CS2"


def test_zero_query_returns_none():
    cache = EmbeddingCache.load(_FakeDB([("CS1", _vec(1, 0, 0))]))
    assert cache.find_best(_vec(0, 0, 0)) is None


def test_multiple_embeddings_per_student_use_best():
    # CS1 has two embeddings; the closer one should win for CS1.
    db = _FakeDB([
        ("CS1", _vec(1, 0, 0)),
        ("CS1", _vec(0, 0, 1)),
        ("CS2", _vec(0, 1, 0)),
    ])
    cache = EmbeddingCache.load(db)
    assert len(cache) == 3
    assert cache.unique_students() == 2
    match = cache.find_best(_vec(0, 0, 1), threshold=0.45)
    assert match is not None and match[0] == "CS1"


def test_reload_embeddings_alias_refreshes():
    db = _FakeDB([("CS1", _vec(1, 0, 0))])
    cache = EmbeddingCache.load(db)
    assert len(cache) == 1
    db._rows.append(("CS2", _vec(0, 1, 0)))
    cache.reload_embeddings(db)
    assert len(cache) == 2


if __name__ == "__main__":
    import sys

    from tests.run_tests import run_module

    sys.exit(0 if run_module(sys.modules[__name__]) else 1)
