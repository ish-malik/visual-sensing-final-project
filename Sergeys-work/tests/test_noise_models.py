"""Unit tests for noise_models CIS and DVS simulators."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))

from noise_models import (
    CisNoiseConfig,
    DvsNoiseConfig,
    simulate_cis_noisy_gt,
    simulate_dvs_noisy_gt,
)


def _fake_gt(num_frames=10, num_objs=3, im_w=1920, im_h=1080):
    rng = np.random.default_rng(0)
    rows = []
    for obj_id in range(1, num_objs + 1):
        x0 = rng.uniform(100, im_w - 200)
        y0 = rng.uniform(100, im_h - 400)
        for f in range(1, num_frames + 1):
            rows.append({
                'frame': f, 'id': obj_id,
                'x': x0 + f * 2, 'y': y0 + f * 1,
                'w': 80, 'h': 200,
                'conf': 1, 'cls': 1, 'vis': 1.0,
            })
    return pd.DataFrame(rows)


def test_cis_no_noise_returns_gt_like():
    """With FP=0 and no noise, output detections should approximate GT
    counts (modulo motion blur position noise)."""
    gt = _fake_gt()
    cfg = CisNoiseConfig(
        actual_fps=30.0, resolution=(1920, 1080), adc_bits=12,
        motion_blur_fraction=0.5, bg_fp_rate_per_frame=0.0,
    )
    out = simulate_cis_noisy_gt(gt, fps=30.0, config=cfg,
                                 rng=np.random.default_rng(0))
    assert len(out) >= len(gt) * 0.9


def test_cis_fp_injection_adds_detections():
    """bg_fp_rate_per_frame > 0 adds extra rows vs FP=0."""
    gt = _fake_gt()
    cfg_clean = CisNoiseConfig(
        actual_fps=30.0, resolution=(1920, 1080), adc_bits=12,
        bg_fp_rate_per_frame=0.0,
    )
    cfg_noisy = CisNoiseConfig(
        actual_fps=30.0, resolution=(1920, 1080), adc_bits=12,
        bg_fp_rate_per_frame=5.0,
    )
    out_clean = simulate_cis_noisy_gt(
        gt, fps=30.0, config=cfg_clean, rng=np.random.default_rng(0),
    )
    out_noisy = simulate_cis_noisy_gt(
        gt, fps=30.0, config=cfg_noisy, rng=np.random.default_rng(0),
    )
    assert len(out_noisy) > len(out_clean) + 30


def test_cis_determinism_with_seed():
    """Same seed -> identical output."""
    gt = _fake_gt()
    cfg = CisNoiseConfig(
        actual_fps=30.0, resolution=(1920, 1080), adc_bits=10,
        bg_fp_rate_per_frame=3.0,
    )
    a = simulate_cis_noisy_gt(gt, 30.0, cfg, np.random.default_rng(42))
    b = simulate_cis_noisy_gt(gt, 30.0, cfg, np.random.default_rng(42))
    pd.testing.assert_frame_equal(a, b)


def test_dvs_coast_toggle_changes_output():
    """With coasting enabled, missed detections get replaced by drift
    of the last bbox. Without it, the detection is simply absent."""
    gt = _fake_gt(num_frames=20)
    cfg_coast = DvsNoiseConfig(
        resolution=(1920, 1080), contrast_threshold=0.10,
        pixel_latency_s=20e-6, refractory_cap=12e6,
        coast=True, coast_frames=10,
        bg_fp_rate_per_frame=0.0,
    )
    cfg_nocoast = DvsNoiseConfig(
        resolution=(1920, 1080), contrast_threshold=0.10,
        pixel_latency_s=20e-6, refractory_cap=12e6,
        coast=False,
        bg_fp_rate_per_frame=0.0,
    )
    out_c  = simulate_dvs_noisy_gt(
        gt, 30.0, cfg_coast,   np.random.default_rng(0),
    )
    out_nc = simulate_dvs_noisy_gt(
        gt, 30.0, cfg_nocoast, np.random.default_rng(0),
    )
    assert len(out_c) >= len(out_nc)


def test_dvs_determinism_with_seed():
    gt = _fake_gt(num_frames=15)
    cfg = DvsNoiseConfig(
        resolution=(1920, 1080), contrast_threshold=0.20,
        pixel_latency_s=20e-6, refractory_cap=12e6,
        coast=False, bg_fp_rate_per_frame=2.0,
    )
    a = simulate_dvs_noisy_gt(gt, 30.0, cfg, np.random.default_rng(7))
    b = simulate_dvs_noisy_gt(gt, 30.0, cfg, np.random.default_rng(7))
    pd.testing.assert_frame_equal(a, b)
