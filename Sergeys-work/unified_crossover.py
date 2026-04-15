"""Analytical CIS vs DVS power sweep that closes Gap 3 of the team proposal.

This module wires Ramaa's scene model to Harshitha's ModuCIS power
numbers and Ish's DVS circuit formula so we can compute total sensor
power across a grid of velocities, object sizes and backgrounds. The
output is the crossover velocity V star at which DVS becomes cheaper
than CIS for a given sensor pair, plus the CSV that feeds the main
crossover figure.

Sweep A is a pure closed form calculation, so it runs in a second with
no seeds or workers. Sweep B, which needs real MOT17 frames and the
noise simulators, lives in run_sweeps.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'Ramaas-work')))

from sensor_database import DVS_SENSORS, CIS_SENSORS, DVSSensor, CISSensor
from visualcomputing import compute_event_rate, compute_fps_min, backgrounds


# Lichtsteiner 2008 nominal contrast threshold, used everywhere as the
# reference point for the 1/theta event rate scaling and the theta squared
# energy scaling. Keeping a single constant here avoids the old repo's
# problem of three files each picking their own baseline.
THETA_REF   = 0.20
DVS_FP_RATE = 0.10    # fraction of events that are false positives
DVS_FN_RATE = 0.05    # fraction of real contrast crossings missed

VELOCITIES_PX_S = [10, 50, 100, 200, 500, 1000, 2000]
OBJECT_SIZES_PX = [25, 50, 100]
BG_DENSITIES    = sorted(set(backgrounds.values()))    # 0.05 and 0.40
THETAS          = [0.05, 0.10, 0.20, 0.40]
CIS_POLICIES    = ['locked', 'adaptive']
SAFETY_FACTOR   = 10


def event_rate_at_theta(velocity_px_s: float, obj_size_px: float,
                         bg_density: float, theta: float,
                         fp_rate: float = 0.10) -> float:
    """Ramaa's scene event rate, scaled by 1 over theta.

    A higher contrast threshold needs more integrated brightness change
    before each event fires, so raw event rate drops roughly as 1 over
    theta. This follows the v2e paper convention with THETA_REF as the
    reference operating point.
    """
    base = compute_event_rate(velocity_px_s, obj_size_px, bg_density, fp_rate)
    return float(base) * (THETA_REF / theta)


def cis_required_fps(velocity_px_s: float, obj_size_px: float, policy: str,
                      worst_case_fps: float, max_fps: float) -> float:
    """FPS the CIS would run at for a given velocity under a given policy.

    Locked always returns the worst case FPS that Harshitha's ModuCIS
    code assumes. Adaptive picks the minimum FPS needed to track the
    current velocity and clamps it to what the sensor can actually do.
    """
    if policy == 'locked':
        return float(worst_case_fps)
    if policy == 'adaptive':
        fps_needed = compute_fps_min(velocity_px_s, obj_size_px, SAFETY_FACTOR)
        return float(min(max_fps, max(1.0, fps_needed)))
    raise ValueError(f'unknown CIS policy: {policy!r}')


def dvs_power_custom(event_rate_ev_s: float, theta: float,
                      sensor: DVSSensor) -> tuple[float, dict]:
    """Total DVS power for a given event rate, threshold and sensor.

    Uses Ish's circuit formula but takes the static floor and the
    reference energy per event straight from the sensor datasheet. The
    energy per event scales as theta squared because the comparator
    feedback capacitor stores energy proportional to the square of its
    voltage swing. Event rate is capped at the refractory limit, and FP
    and FN fractions are counted the same way Ish's model does.

    Returns the total power in mW and a breakdown dict.
    """
    e_per_event_nj = sensor.e_per_event_nj * (theta / THETA_REF) ** 2
    cap_rate = sensor.max_event_rate_mevps * 1e6
    eff_rate = min(event_rate_ev_s, cap_rate)
    saturated = event_rate_ev_s > cap_rate

    r_fp    = DVS_FP_RATE * eff_rate
    r_fn    = DVS_FN_RATE * eff_rate
    r_valid = eff_rate - r_fn

    dyn_valid_mw = r_valid * e_per_event_nj * 1e-9 * 1e3
    dyn_fp_mw    = r_fp    * e_per_event_nj * 1e-9 * 1e3
    p_total = sensor.p_static_mw + dyn_valid_mw + dyn_fp_mw

    return p_total, {
        'e_per_event_nj':  round(e_per_event_nj, 6),
        'eff_rate_ev_s':   round(eff_rate, 1),
        'raw_rate_ev_s':   round(event_rate_ev_s, 1),
        'saturated':       bool(saturated),
        'p_static_mw':     sensor.p_static_mw,
        'p_dyn_valid_mw':  round(dyn_valid_mw, 6),
        'p_dyn_fp_mw':     round(dyn_fp_mw, 6),
        'p_total_mw':      round(p_total, 6),
    }


def cis_power_custom(velocity_px_s: float, obj_size_px: float,
                      sensor: CISSensor, policy: str, worst_case_fps: float,
                      modulcis_lut=None) -> tuple[float, dict]:
    """Total CIS power for a given velocity and policy.

    By default this interpolates linearly between the sensor's idle power
    and its power at max FPS. If a ModuCIS LUT is passed in it queries
    that instead, which is slower but matches the SPICE model directly.
    """
    fps = cis_required_fps(velocity_px_s, obj_size_px, policy,
                           worst_case_fps, sensor.max_fps)
    if modulcis_lut is not None:
        p_mw = modulcis_lut.query(resolution=sensor.resolution,
                                   fps=fps, adc_bits=sensor.adc_bits)
    else:
        p_mw = sensor.power_mw(fps)
    return float(p_mw), {
        'fps_used':         round(fps, 3),
        'policy':           policy,
        'worst_case_fps':   round(worst_case_fps, 3),
        'resolution':       sensor.resolution,
        'adc_bits':         sensor.adc_bits,
    }


def run_sweep_a(out_csv: str, worst_case_fps: float | None = None,
                 modulcis_lut=None) -> pd.DataFrame:
    """Run the closed form sweep and write the CSV.

    The grid is 7 velocities by 3 object sizes by 2 backgrounds, crossed
    with 4 CIS sensors at 2 policies and 4 DVS sensors at 4 thresholds,
    which gives 1008 rows. Each row is a single arithmetic call so the
    whole sweep finishes in about a second.
    """
    rows = []

    if worst_case_fps is None:
        # Derive from Ramaa's fastest object / smallest size:
        worst_case_fps = compute_fps_min(
            max(VELOCITIES_PX_S), min(OBJECT_SIZES_PX), SAFETY_FACTOR,
        )

    for velocity in VELOCITIES_PX_S:
        for obj_size in OBJECT_SIZES_PX:
            for bg_density in BG_DENSITIES:
                # CIS sweep: 4 sensors x 2 policies
                for cis_sensor in CIS_SENSORS:
                    for policy in CIS_POLICIES:
                        p_mw, breakdown = cis_power_custom(
                            velocity, obj_size, cis_sensor,
                            policy, worst_case_fps, modulcis_lut,
                        )
                        rows.append({
                            'sensor_name':        cis_sensor.name,
                            'sensor_type':        'CIS',
                            'velocity_px_s':      velocity,
                            'obj_size_px':        obj_size,
                            'bg_density':         bg_density,
                            'bg_label':           _bg_label(bg_density),
                            'operating_point':    policy,
                            'theta':              None,
                            'fps_used':           breakdown['fps_used'],
                            'event_rate_ev_s':    None,
                            'power_mw':           round(p_mw, 4),
                            'power_static_mw':    None,
                            'power_dynamic_mw':   None,
                            'saturated':          False,
                            'resolution':         f'{cis_sensor.resolution[0]}x{cis_sensor.resolution[1]}',
                            'sensor_price':       cis_sensor.price_usd,
                        })

                # DVS sweep: 4 sensors x 4 thetas
                for dvs_sensor in DVS_SENSORS:
                    for theta in THETAS:
                        ev_rate = event_rate_at_theta(
                            velocity, obj_size, bg_density, theta,
                        )
                        p_mw, breakdown = dvs_power_custom(
                            ev_rate, theta, dvs_sensor,
                        )
                        rows.append({
                            'sensor_name':        dvs_sensor.name,
                            'sensor_type':        'DVS',
                            'velocity_px_s':      velocity,
                            'obj_size_px':        obj_size,
                            'bg_density':         bg_density,
                            'bg_label':           _bg_label(bg_density),
                            'operating_point':    f'theta={theta:.2f}',
                            'theta':              theta,
                            'fps_used':           None,
                            'event_rate_ev_s':    round(ev_rate, 1),
                            'power_mw':           round(p_mw, 4),
                            'power_static_mw':    breakdown['p_static_mw'],
                            'power_dynamic_mw':   round(
                                breakdown['p_dyn_valid_mw']
                                + breakdown['p_dyn_fp_mw'], 6),
                            'saturated':          breakdown['saturated'],
                            'resolution':         f'{dvs_sensor.resolution[0]}x{dvs_sensor.resolution[1]}',
                            'sensor_price':       dvs_sensor.price_usd,
                        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def _bg_label(bg_density: float) -> str:
    for name, val in backgrounds.items():
        if abs(val - bg_density) < 1e-6:
            return name
    return f'bg={bg_density:.2f}'


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(
        HERE, 'results', 'sweep_a_analytical.csv'))
    args = ap.parse_args()

    print(f'[Sweep A] running analytical crossover sweep...')
    df = run_sweep_a(args.out)
    print(f'[Sweep A] wrote {len(df)} rows to {args.out}')
    print(f'[Sweep A] first 5 rows:')
    print(df.head().to_string())
    print()
    print(f'[Sweep A] sensor counts:')
    print(df.groupby(['sensor_type', 'operating_point']).size())
