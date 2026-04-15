"""Unit tests for metrics_hota.

Tests are against hand-verified cases: perfect tracking, total miss,
partial match with ID switches.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))

from metrics_hota import compute_hota


def _make_df(rows):
    return pd.DataFrame(rows, columns=['frame', 'id', 'x', 'y', 'w', 'h'])


def test_perfect_tracking_is_hota_one():
    """Predictions identical to GT -> HOTA == 1."""
    gt = _make_df([
        (1, 1, 0, 0, 10, 20),
        (1, 2, 50, 50, 10, 20),
        (2, 1, 1, 0, 10, 20),
        (2, 2, 51, 50, 10, 20),
    ])
    result = compute_hota(gt.copy(), gt.copy())
    assert result['hota'] == pytest.approx(1.0, abs=1e-6)
    assert result['det_a'] == pytest.approx(1.0, abs=1e-6)
    assert result['ass_a'] == pytest.approx(1.0, abs=1e-6)


def test_empty_predictions_is_hota_zero():
    """No predictions at all -> HOTA == 0."""
    gt = _make_df([(1, 1, 0, 0, 10, 20)])
    pr = _make_df([])
    result = compute_hota(pr, gt)
    assert result['hota'] == pytest.approx(0.0, abs=1e-6)
    assert result['det_a'] == pytest.approx(0.0, abs=1e-6)


def test_all_false_positives_is_hota_zero():
    """Predictions that don't overlap any GT -> HOTA == 0."""
    gt = _make_df([(1, 1, 0, 0, 10, 20)])
    pr = _make_df([(1, 99, 500, 500, 10, 20)])
    result = compute_hota(pr, gt)
    assert result['hota'] == pytest.approx(0.0, abs=1e-6)


def test_id_switch_reduces_ass_a_not_det_a():
    """
    Same detections every frame, but the ID flips between frames 1 and 2.
    DetA should be 1.0 (every GT is matched), AssA should be < 1
    (two distinct pred IDs, each matches the GT only half the time).
    """
    gt = _make_df([
        (1, 1, 0, 0, 10, 20),
        (2, 1, 0, 0, 10, 20),
    ])
    pr = _make_df([
        (1, 10, 0, 0, 10, 20),
        (2, 20, 0, 0, 10, 20),
    ])
    result = compute_hota(pr, gt)
    assert result['det_a'] == pytest.approx(1.0, abs=0.05)
    assert result['ass_a'] < 0.9
    assert result['hota'] < 0.95


def test_hota_bounded_zero_to_one():
    """HOTA, DetA, AssA all in [0, 1]."""
    np.random.seed(0)
    gt = _make_df([
        (f, i, np.random.uniform(0, 100), np.random.uniform(0, 100), 10, 20)
        for f in range(1, 6) for i in range(1, 4)
    ])
    pr = _make_df([
        (f, i, np.random.uniform(0, 100), np.random.uniform(0, 100), 10, 20)
        for f in range(1, 6) for i in range(1, 4)
    ])
    result = compute_hota(pr, gt)
    for k in ('hota', 'det_a', 'ass_a'):
        assert 0.0 <= result[k] <= 1.0, f'{k} out of range: {result[k]}'
