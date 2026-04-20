"""Sweep A: closed-form CIS vs DVS power grid to find v*."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)  # Sergeys-work/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_ROOT, "..", "Ramaas-work")))

from models.sensor_database import DVS_SENSORS, CIS_SENSORS, DVSSensor, CISSensor
from visualcomputing import compute_event_rate, compute_fps_min, backgrounds


# baseline for 1/theta rate and theta^2 energy scaling.
THETA_REF = 0.20
DVS_FP_RATE = 0.10
DVS_FN_RATE = 0.05

VELOCITIES_PX_S = [10, 50, 100, 200, 500, 1000, 2000]
OBJECT_SIZES_PX = [25, 50, 100]
BG_DENSITIES = [0.05, 0.10, 0.20, 0.30, 0.40]
THETAS = [0.05, 0.10, 0.20, 0.40]
CIS_POLICIES = ["locked", "adaptive"]
SAFETY_FACTOR = 10


def event_rate_at_theta(
    velocity_px_s: float,
    obj_size_px: float,
    bg_density: float,
    theta: float,
    fp_rate: float = 0.10,
) -> float:
    """Event rate scaled to theta (v2e 1/theta convention)."""
    base = compute_event_rate(velocity_px_s, obj_size_px, bg_density, fp_rate)
    return float(base) * (THETA_REF / theta)


def cis_required_fps(
    velocity_px_s: float,
    obj_size_px: float,
    policy: str,
    worst_case_fps: float,
    max_fps: float,
) -> float:
    """Locked = worst case. Adaptive = whatever the motion actually needs."""
    if policy == "locked":
        return float(worst_case_fps)
    if policy == "adaptive":
        fps_needed = compute_fps_min(velocity_px_s, obj_size_px, SAFETY_FACTOR)
        return float(min(max_fps, max(1.0, fps_needed)))
    raise ValueError(f"unknown CIS policy: {policy!r}")


def dvs_power_custom(
    event_rate_ev_s: float, theta: float, sensor: DVSSensor
) -> tuple[float, dict]:
    """DVS total power (mW) + breakdown. Energy/event ~ theta^2."""
    e_per_event_nj = sensor.e_per_event_nj * (theta / THETA_REF) ** 2
    cap_rate = sensor.max_event_rate_mevps * 1e6
    eff_rate = min(event_rate_ev_s, cap_rate)
    saturated = event_rate_ev_s > cap_rate

    r_fp = DVS_FP_RATE * eff_rate
    r_fn = DVS_FN_RATE * eff_rate
    r_valid = eff_rate - r_fn

    dyn_valid_mw = r_valid * e_per_event_nj * 1e-9 * 1e3
    dyn_fp_mw = r_fp * e_per_event_nj * 1e-9 * 1e3
    p_total = sensor.p_static_mw + dyn_valid_mw + dyn_fp_mw

    return p_total, {
        "e_per_event_nj": round(e_per_event_nj, 6),
        "eff_rate_ev_s": round(eff_rate, 1),
        "raw_rate_ev_s": round(event_rate_ev_s, 1),
        "saturated": bool(saturated),
        "p_static_mw": sensor.p_static_mw,
        "p_dyn_valid_mw": round(dyn_valid_mw, 6),
        "p_dyn_fp_mw": round(dyn_fp_mw, 6),
        "p_total_mw": round(p_total, 6),
    }


def cis_power_custom(
    velocity_px_s: float,
    obj_size_px: float,
    sensor: CISSensor,
    policy: str,
    worst_case_fps: float,
    spice_lut=None,
) -> tuple[float, dict]:
    """CIS total power. Datasheet interp, or SPICE LUT if given."""
    fps = cis_required_fps(
        velocity_px_s, obj_size_px, policy, worst_case_fps, sensor.max_fps
    )
    if spice_lut is not None:
        p_mw = spice_lut.query(
            resolution=sensor.resolution, fps=fps, adc_bits=sensor.adc_bits
        )
    else:
        p_mw = sensor.power_mw(fps)
    return float(p_mw), {
        "fps_used": round(fps, 3),
        "policy": policy,
        "worst_case_fps": round(worst_case_fps, 3),
        "resolution": sensor.resolution,
        "adc_bits": sensor.adc_bits,
    }


def run_sweep_a(
    out_csv: str, worst_case_fps: float | None = None, spice_lut=None
) -> pd.DataFrame:
    """Walk the grid, write the CSV."""
    rows = []

    if worst_case_fps is None:
        # fastest motion, smallest object
        worst_case_fps = compute_fps_min(
            max(VELOCITIES_PX_S),
            min(OBJECT_SIZES_PX),
            SAFETY_FACTOR,
        )

    for velocity in VELOCITIES_PX_S:
        for obj_size in OBJECT_SIZES_PX:
            for bg_density in BG_DENSITIES:
                for cis_sensor in CIS_SENSORS:
                    for policy in CIS_POLICIES:
                        rows.append(
                            _cis_row(
                                cis_sensor,
                                velocity,
                                obj_size,
                                bg_density,
                                policy,
                                worst_case_fps,
                                spice_lut,
                            )
                        )
                for dvs_sensor in DVS_SENSORS:
                    for theta in THETAS:
                        rows.append(
                            _dvs_row(dvs_sensor, velocity, obj_size, bg_density, theta)
                        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def _res_str(sensor) -> str:
    return f"{sensor.resolution[0]}x{sensor.resolution[1]}"


def _bg_label(bg_density: float) -> str:
    return f"d={bg_density:.2f}"


def _base_row(
    sensor, sensor_type: str, velocity: float, obj_size: float, bg_density: float
) -> dict:
    return {
        "sensor_name": sensor.name,
        "sensor_type": sensor_type,
        "velocity_px_s": velocity,
        "obj_size_px": obj_size,
        "bg_density": bg_density,
        "bg_label": _bg_label(bg_density),
        "resolution": _res_str(sensor),
        "sensor_price": sensor.price_usd,
    }


def _cis_row(
    sensor: CISSensor,
    velocity: float,
    obj_size: float,
    bg_density: float,
    policy: str,
    worst_case_fps: float,
    spice_lut,
) -> dict:
    p_mw, breakdown = cis_power_custom(
        velocity,
        obj_size,
        sensor,
        policy,
        worst_case_fps,
        spice_lut,
    )
    return {
        **_base_row(sensor, "CIS", velocity, obj_size, bg_density),
        "operating_point": policy,
        "theta": None,
        "fps_used": breakdown["fps_used"],
        "event_rate_ev_s": None,
        "power_mw": round(p_mw, 4),
        "power_static_mw": None,
        "power_dynamic_mw": None,
        "saturated": False,
    }


def _dvs_row(
    sensor: DVSSensor, velocity: float, obj_size: float, bg_density: float, theta: float
) -> dict:
    ev_rate = event_rate_at_theta(velocity, obj_size, bg_density, theta)
    p_mw, breakdown = dvs_power_custom(ev_rate, theta, sensor)
    return {
        **_base_row(sensor, "DVS", velocity, obj_size, bg_density),
        "operating_point": f"theta={theta:.2f}",
        "theta": theta,
        "fps_used": None,
        "event_rate_ev_s": round(ev_rate, 1),
        "power_mw": round(p_mw, 4),
        "power_static_mw": breakdown["p_static_mw"],
        "power_dynamic_mw": round(
            breakdown["p_dyn_valid_mw"] + breakdown["p_dyn_fp_mw"], 6
        ),
        "saturated": breakdown["saturated"],
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=os.path.join(HERE, "results", "sweep_a_analytical.csv")
    )
    args = ap.parse_args()

    print(f"[Sweep A] running analytical crossover sweep...")
    df = run_sweep_a(args.out)
    print(f"[Sweep A] wrote {len(df)} rows to {args.out}")
    print(f"[Sweep A] first 5 rows:")
    print(df.head().to_string())
    print()
    print(f"[Sweep A] sensor counts:")
    print(df.groupby(["sensor_type", "operating_point"]).size())
