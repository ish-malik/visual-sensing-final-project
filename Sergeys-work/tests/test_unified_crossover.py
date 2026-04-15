"""Unit tests for unified_crossover power functions."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))

from sensor_database import DVS_SENSORS, CIS_SENSORS
from unified_crossover import (
    dvs_power_custom,
    cis_power_custom,
    cis_required_fps,
    event_rate_at_theta,
    THETA_REF,
)


def _find_dvs(name):
    return next(s for s in DVS_SENSORS if name in s.name)


def _find_cis(name):
    return next(s for s in CIS_SENSORS if name in s.name)


def test_dvs_power_at_theta_ref_roughly_matches_sensor_formula():
    """At theta=THETA_REF the e_per_event scale is 1, so power should
    roughly match sensor.power_mw (minor difference from FP bookkeeping)."""
    s = _find_dvs('Samsung')
    event_rate = 100_000.0
    p_custom, _ = dvs_power_custom(event_rate, THETA_REF, s)
    p_direct = s.power_mw(event_rate)
    assert abs(p_custom - p_direct) / p_direct < 0.10


def test_dvs_power_scales_theta_squared():
    """Energy per event scales as theta^2 -> dynamic power scales as theta^2."""
    s = _find_dvs('Lichtsteiner')  # highest dynamic fraction
    event_rate = 1_000_000.0
    _, bd_low  = dvs_power_custom(event_rate, 0.10, s)
    _, bd_high = dvs_power_custom(event_rate, 0.20, s)
    assert bd_high['e_per_event_nj'] == pytest.approx(
        bd_low['e_per_event_nj'] * 4.0, rel=0.01)


def test_dvs_power_static_floor():
    """At zero event rate, total power == p_static_mw."""
    for s in DVS_SENSORS:
        p, _ = dvs_power_custom(0.0, 0.20, s)
        assert p == pytest.approx(s.p_static_mw, abs=1e-6)


def test_dvs_power_saturates_at_refractory_cap():
    """Event rate above max_event_rate_mevps should be clamped."""
    s = _find_dvs('Lichtsteiner')
    cap = s.max_event_rate_mevps * 1e6
    p_at_cap, _ = dvs_power_custom(cap, 0.20, s)
    p_over, bd = dvs_power_custom(cap * 10, 0.20, s)
    assert p_over == pytest.approx(p_at_cap, abs=1e-6)
    assert bd['saturated']


def test_cis_required_fps_locked_vs_adaptive():
    """Locked returns worst_case; adaptive returns per-velocity minimum."""
    worst = 400.0
    max_fps = 120.0
    # locked: always 400
    assert cis_required_fps(10.0, 50, 'locked', worst, max_fps) == 400.0
    assert cis_required_fps(2000.0, 50, 'locked', worst, max_fps) == 400.0
    # adaptive: velocity/obj*10, clamped to max_fps
    assert cis_required_fps(10.0, 50, 'adaptive', worst, max_fps) == 2.0
    assert cis_required_fps(2000.0, 50, 'adaptive', worst, max_fps) == 120.0


def test_cis_power_locked_is_constant():
    """Under locked policy, CIS power is independent of velocity."""
    s = _find_cis('IMX327')
    worst = 60.0
    p_low, _  = cis_power_custom(10.0, 50, s, 'locked', worst)
    p_high, _ = cis_power_custom(2000.0, 50, s, 'locked', worst)
    assert p_low == pytest.approx(p_high, abs=1e-6)


def test_cis_power_adaptive_grows_with_velocity():
    """Under adaptive policy, slow scenes use less power."""
    s = _find_cis('IMX327')
    worst = 60.0
    p_slow, _ = cis_power_custom(10.0, 50, s, 'adaptive', worst)
    p_fast, _ = cis_power_custom(2000.0, 50, s, 'adaptive', worst)
    assert p_slow < p_fast


def test_event_rate_theta_inverse():
    """event_rate_at_theta is inversely proportional to theta."""
    base = event_rate_at_theta(200, 50, 0.05, THETA_REF)
    double_theta = event_rate_at_theta(200, 50, 0.05, 2 * THETA_REF)
    half_theta = event_rate_at_theta(200, 50, 0.05, 0.5 * THETA_REF)
    assert double_theta == pytest.approx(base * 0.5, rel=1e-6)
    assert half_theta == pytest.approx(base * 2.0, rel=1e-6)
