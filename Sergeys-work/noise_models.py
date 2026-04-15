"""CIS and DVS noise simulators used by the MOT17 validation sweep.

Both functions take ground truth bboxes and return noisy detections.
The CIS path drops frames according to the requested FPS, adds motion
blur and quantisation noise based on ADC bit depth and resolution, and
injects a Poisson stream of background false positives. The DVS path
runs the same kind of noise pass but uses a refractory cap and a minimum
events per frame cutoff, and the coasting behaviour is controlled by an
explicit flag on the config so the comparison figure can toggle it.

All the knobs live on the two config dataclasses. No hidden constants.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from unified_crossover import THETA_REF


@dataclass
class CisNoiseConfig:
    """CIS noise knobs. Defaults are calibrated from MOT17 MOG2 plus SORT runs."""
    actual_fps:              float
    resolution:              tuple[int, int]
    adc_bits:                int
    motion_blur_fraction:    float = 0.5  # share of frame used for integration
    bg_fp_rate_per_frame:    float = 3.0  # Poisson rate for background FPs
    fp_bbox_w_mean:          float = 80.0
    fp_bbox_h_mean:          float = 180.0
    fp_bbox_sigma_pct:       float = 0.25
    displacement_miss_ratio: float = 2.0  # miss when per frame disp exceeds this times width


@dataclass
class DvsNoiseConfig:
    """DVS noise knobs. Defaults roughly match Samsung Gen3.1."""
    resolution:            tuple[int, int]
    contrast_threshold:    float = 0.20
    pixel_latency_s:       float = 20e-6
    refractory_cap:        float = 12e6   # max events per second across array
    min_velocity_px_s:     float = 5.0    # static objects fire no events
    shot_noise_rate_hz:    float = 1.0    # from Ish's v2e calibration
    leak_rate_hz:          float = 0.1
    mismatch_sigma:        float = 0.03
    bg_fp_rate_per_frame:  float = 3.0
    fp_bbox_w_mean:        float = 80.0
    fp_bbox_h_mean:        float = 180.0
    fp_bbox_sigma_pct:     float = 0.25
    coast:                 bool  = False
    coast_frames:          int   = 150
    coast_drift_px:        float = 0.3


def _velocity_per_object(gt: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Tag each GT row with the object's euclidean velocity."""
    gt = gt.sort_values(['id', 'frame']).copy()
    gt['cx'] = gt['x'] + gt['w'] / 2
    gt['cy'] = gt['y'] + gt['h'] / 2
    gt['prev_cx'] = gt.groupby('id')['cx'].shift(1)
    gt['prev_cy'] = gt.groupby('id')['cy'].shift(1)
    dx = gt['cx'] - gt['prev_cx']
    dy = gt['cy'] - gt['prev_cy']
    gt['displacement_px'] = np.sqrt(dx**2 + dy**2).fillna(0.0)
    gt['velocity_px_s']   = (gt['displacement_px'] * fps).fillna(0.0)
    return gt


def _inject_fps(det_rows: list, frame: int, im_w: int, im_h: int,
                lambda_per_frame: float,
                fp_bbox_w_mean: float, fp_bbox_h_mean: float,
                fp_bbox_sigma_pct: float, rng: np.random.Generator,
                next_id: int) -> int:
    """Add a Poisson number of false positive boxes to this frame."""
    n = rng.poisson(lambda_per_frame)
    for _ in range(n):
        w = max(4.0, fp_bbox_w_mean * (1.0 + rng.normal(0, fp_bbox_sigma_pct)))
        h = max(4.0, fp_bbox_h_mean * (1.0 + rng.normal(0, fp_bbox_sigma_pct)))
        x = rng.uniform(0, max(1, im_w - w))
        y = rng.uniform(0, max(1, im_h - h))
        det_rows.append({
            'frame': frame, 'id': next_id,
            'x': x, 'y': y, 'w': w, 'h': h,
        })
        next_id += 1
    return next_id


def simulate_cis_noisy_gt(gt_df: pd.DataFrame, fps: float,
                           config: CisNoiseConfig,
                           rng: np.random.Generator) -> pd.DataFrame:
    """Turn GT bboxes into what a CIS detector would output on those frames.

    Drops frames to match the requested FPS, kills detections where the
    object moves too far between frames, adds motion blur and ADC
    quantisation noise to position and size, and finishes by injecting
    background false positives.
    """
    REFERENCE_WIDTH   = 1920
    sensor_width      = config.resolution[0]
    spatial_scale     = REFERENCE_WIDTH / sensor_width
    resolution_sigma  = 0.5 * spatial_scale

    adc_noise_factor  = 2.0 ** (12 - config.adc_bits)
    adc_noise_sigma   = 0.3 * adc_noise_factor
    adc_miss_prob     = 0.02 * max(0.0, adc_noise_factor - 1)

    gt = _velocity_per_object(gt_df, fps)

    all_frames = sorted(gt['frame'].unique())
    if config.actual_fps < fps:
        step = max(1, int(round(fps / config.actual_fps)))
        observed_frames = set(all_frames[::step])
    else:
        observed_frames = set(all_frames)

    effective_fps = min(config.actual_fps, fps)
    sensor_dt     = 1.0 / effective_fps

    obs = gt[gt['frame'].isin(observed_frames)].copy()
    if obs.empty:
        return pd.DataFrame(columns=['frame', 'id', 'x', 'y', 'w', 'h'])

    velocities = obs['velocity_px_s'].values
    widths     = obs['w'].values
    heights    = obs['h'].values

    # Displacement miss
    disp = velocities * sensor_dt
    keep = disp <= config.displacement_miss_ratio * widths
    if adc_miss_prob > 0:
        keep &= rng.random(len(obs)) >= adc_miss_prob

    obs = obs[keep].copy()
    if obs.empty:
        out = pd.DataFrame(columns=['frame', 'id', 'x', 'y', 'w', 'h'])
    else:
        velocities = obs['velocity_px_s'].values
        widths     = obs['w'].values
        heights    = obs['h'].values

        blur_sigma = velocities * (config.motion_blur_fraction * sensor_dt)
        total_pos_sigma = np.sqrt(
            blur_sigma**2 + resolution_sigma**2 + adc_noise_sigma**2)
        total_pos_sigma = np.maximum(total_pos_sigma, 0.3)

        noise_x = rng.normal(0, total_pos_sigma)
        noise_y = rng.normal(0, total_pos_sigma)
        size_sigma = 0.03 + 0.02 * (spatial_scale - 1)
        noise_w = rng.normal(0, size_sigma * widths)
        noise_h = rng.normal(0, size_sigma * heights)

        out = pd.DataFrame({
            'frame': obs['frame'].values,
            'id':    obs['id'].values.astype(int),
            'x':     obs['x'].values + noise_x,
            'y':     obs['y'].values + noise_y,
            'w':     np.maximum(1, widths  + noise_w),
            'h':     np.maximum(1, heights + noise_h),
        })

    # FP injection (uses "next_id" beyond max GT id)
    if config.bg_fp_rate_per_frame > 0.0:
        im_h_ref = int(gt['y'].max() + gt['h'].max())
        im_w_ref = int(gt['x'].max() + gt['w'].max())
        fp_rows = []
        next_id = int(gt['id'].max()) + 1_000_000 if len(gt) else 1
        for frame in sorted(observed_frames):
            next_id = _inject_fps(
                fp_rows, frame, im_w_ref, im_h_ref,
                config.bg_fp_rate_per_frame,
                config.fp_bbox_w_mean, config.fp_bbox_h_mean,
                config.fp_bbox_sigma_pct, rng, next_id,
            )
        if fp_rows:
            fp_df = pd.DataFrame(fp_rows)
            out = pd.concat([out, fp_df], ignore_index=True)

    return out.sort_values(['frame', 'id']).reset_index(drop=True)


def simulate_dvs_noisy_gt(gt_df: pd.DataFrame, fps: float,
                           config: DvsNoiseConfig,
                           rng: np.random.Generator) -> pd.DataFrame:
    """Turn GT bboxes into what a DVS detector would output on those frames.

    Drops static objects, then drops detections that hit the refractory
    cap or fall below the minimum events per frame threshold. Adds
    latency and resolution noise to the survivors. When coast is on, a
    missed detection keeps emitting the last known bbox with a bit of
    drift until coast_frames have passed. Finally injects background
    false positives the same way the CIS path does.
    """
    REFERENCE_WIDTH  = 1920
    sensor_width     = config.resolution[0]
    spatial_scale    = REFERENCE_WIDTH / sensor_width
    resolution_sigma = 0.5 * spatial_scale

    MIN_EVENTS_PER_FRAME = 50

    # Scale factor converts Ramaa's rate to this theta (not used for detection
    # here -- detection is modelled via refractory saturation + min events).
    theta_scale = THETA_REF / max(config.contrast_threshold, 1e-6)

    latency_factor = np.sqrt(config.pixel_latency_s / 1e-6) * 0.05
    size_scale     = latency_factor + 0.02 * max(0.0, spatial_scale - 1)

    gt = _velocity_per_object(gt_df, fps)
    num_rows = len(gt)
    velocities = gt['velocity_px_s'].values
    widths     = gt['w'].values
    heights    = gt['h'].values

    is_moving = velocities >= config.min_velocity_px_s

    perimeter   = 2 * (widths + heights)
    event_rate  = perimeter * velocities * theta_scale
    eff_rate    = np.minimum(event_rate, config.refractory_cap)

    sat_miss_prob = np.where(
        event_rate > config.refractory_cap,
        1.0 - config.refractory_cap / np.maximum(event_rate, 1),
        0.0,
    )
    sat_pass = rng.random(num_rows) >= sat_miss_prob

    events_per_frame = eff_rate / fps
    sparse_miss_prob = np.where(
        events_per_frame < MIN_EVENTS_PER_FRAME,
        1.0 - events_per_frame / MIN_EVENTS_PER_FRAME,
        0.0,
    )
    sparse_pass = rng.random(num_rows) >= sparse_miss_prob

    detected = is_moving & sat_pass & sparse_pass

    latency_sigma = velocities * config.pixel_latency_s
    pos_sigma = np.sqrt(latency_sigma**2 + resolution_sigma**2)
    pos_sigma = np.maximum(pos_sigma, 0.1)

    noise_x = rng.normal(0, pos_sigma)
    noise_y = rng.normal(0, pos_sigma)
    noise_w = rng.normal(0, size_scale * widths)
    noise_h = rng.normal(0, size_scale * heights)

    det_x = gt['x'].values + noise_x
    det_y = gt['y'].values + noise_y
    det_w = np.maximum(1, widths  + noise_w)
    det_h = np.maximum(1, heights + noise_h)

    frame_numbers = gt['frame'].values
    object_ids    = gt['id'].values
    unique_frames = sorted(gt['frame'].unique())

    out_rows = []

    last_bbox:         dict[int, tuple[float, float, float, float]] = {}
    frames_since_det:  dict[int, int] = {}

    for frame in unique_frames:
        mask       = frame_numbers == frame
        frame_idx  = np.where(mask)[0]
        active_ids = set()

        for row_idx in frame_idx:
            object_id = int(object_ids[row_idx])
            active_ids.add(object_id)

            if detected[row_idx]:
                bx, by = det_x[row_idx], det_y[row_idx]
                bw, bh = det_w[row_idx], det_h[row_idx]
                out_rows.append({
                    'frame': frame, 'id': object_id,
                    'x': bx, 'y': by, 'w': bw, 'h': bh,
                })
                last_bbox[object_id]        = (bx, by, bw, bh)
                frames_since_det[object_id] = 0
            elif (config.coast and object_id in last_bbox
                  and frames_since_det.get(object_id, config.coast_frames)
                      < config.coast_frames):
                bx, by, bw, bh = last_bbox[object_id]
                coast_age = frames_since_det[object_id] + 1
                drift = coast_age * config.coast_drift_px
                out_rows.append({
                    'frame': frame, 'id': object_id,
                    'x': bx + rng.normal(0, drift),
                    'y': by + rng.normal(0, drift),
                    'w': bw, 'h': bh,
                })
                frames_since_det[object_id] = coast_age
            else:
                # no detection, no coast -> object invisible this frame
                if object_id in frames_since_det:
                    frames_since_det[object_id] = frames_since_det.get(
                        object_id, 0) + 1

        for object_id in list(last_bbox.keys()):
            if object_id not in active_ids:
                del last_bbox[object_id]
                frames_since_det.pop(object_id, None)

    out = (pd.DataFrame(out_rows)
           if out_rows
           else pd.DataFrame(columns=['frame', 'id', 'x', 'y', 'w', 'h']))

    # FP injection
    if config.bg_fp_rate_per_frame > 0.0 and len(gt_df):
        im_h_ref = int(gt_df['y'].max() + gt_df['h'].max())
        im_w_ref = int(gt_df['x'].max() + gt_df['w'].max())
        fp_rows = []
        next_id = int(gt_df['id'].max()) + 1_000_000
        for frame in unique_frames:
            next_id = _inject_fps(
                fp_rows, frame, im_w_ref, im_h_ref,
                config.bg_fp_rate_per_frame,
                config.fp_bbox_w_mean, config.fp_bbox_h_mean,
                config.fp_bbox_sigma_pct, rng, next_id,
            )
        if fp_rows:
            fp_df = pd.DataFrame(fp_rows)
            out = pd.concat([out, fp_df], ignore_index=True)

    return out.sort_values(['frame', 'id']).reset_index(drop=True)
