"""Runs Sweep A and Sweep B and writes the two result CSVs to results/.

Sweep A is just a forward call to unified_crossover.run_sweep_a, which is
closed form and finishes in under a second. Sweep B runs each of the 240
sensor and operating point configurations through the noise simulators,
the SORT tracker and the HOTA metrics, spread across a process pool.
At 1050 frames with 5 seeds this takes about 9 minutes on 16 workers.
The DVS half of Sweep B runs every config twice, once with coasting off
and once with it on, so the coast comparison figure has data to plot.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sensor_database import DVS_SENSORS, CIS_SENSORS
from unified_crossover import run_sweep_a


MOT17_SEQ_DIR = os.path.join(HERE, 'MOT17', 'train', 'MOT17-04-SDP')
OUT_DIR       = os.path.join(HERE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def _run_one_sweep_b(config: dict) -> dict:
    """One sensor, operating point, coast flag, seed run. Pickleable for the pool."""
    import os as _os
    import sys as _sys
    here = config['here']
    _sys.path.insert(0, here)

    import numpy as _np

    from ingest_mot import load_seqinfo, load_gt
    from fast_sort import Sort
    from evaluate_tracking import tracks_to_df, evaluate as evaluate_mota
    from metrics_hota import compute_hota
    from noise_models import (
        CisNoiseConfig, DvsNoiseConfig,
        simulate_cis_noisy_gt, simulate_dvs_noisy_gt,
    )

    info  = load_seqinfo(config['seq_dir'])
    gt_df = load_gt(info)
    gt_sub = gt_df[gt_df['frame'] <= config['max_frames']].copy()

    rng = _np.random.default_rng(config['seed'])

    if config['sensor_type'] == 'DVS':
        cfg = DvsNoiseConfig(**config['cfg_kwargs'])
        dets = simulate_dvs_noisy_gt(gt_sub, info.frame_rate, cfg, rng)
    else:
        cfg = CisNoiseConfig(**config['cfg_kwargs'])
        dets = simulate_cis_noisy_gt(gt_sub, info.frame_rate, cfg, rng)

    # Feed detections into SORT. Group by frame for fast lookup.
    det_by_frame: dict[int, _np.ndarray] = {}
    if len(dets):
        for frame_key, grp in dets.groupby('frame'):
            det_by_frame[int(frame_key)] = grp[['x', 'y', 'w', 'h']].to_numpy()

    tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3, color_gate=0.0)
    tracks_per_frame = []
    for frame_num in range(1, config['max_frames'] + 1):
        frame_dets = det_by_frame.get(frame_num, _np.empty((0, 4)))
        tracks = tracker.update(frame_dets)
        tracks_per_frame.append((frame_num, tracks))

    pred_df = tracks_to_df(tracks_per_frame)
    hota = compute_hota(pred_df, gt_sub)
    mota_metrics = evaluate_mota(pred_df, gt_sub)

    return {
        **{k: v for k, v in config.items()
           if k in ('sensor_name', 'sensor_type', 'operating_point',
                    'theta', 'fps_used', 'coast', 'seed', 'power_mw')},
        'hota':          round(hota['hota'], 4),
        'det_a':         round(hota['det_a'], 4),
        'ass_a':         round(hota['ass_a'], 4),
        'mota':          round(mota_metrics['mota'], 4),
        'idf1':          round(mota_metrics['idf1'], 4),
        'id_switches':   int(mota_metrics['id_switches']),
        'num_pred':      int(mota_metrics['num_pred']),
        'num_det_input': int(len(dets)),
    }


def _build_sweep_b_configs(max_frames: int, num_seeds: int,
                            fp_calibration: dict) -> list[dict]:
    configs = []
    dvs_thetas = [0.05, 0.10, 0.20, 0.40]
    cis_fps_grid = [5, 15, 30, None]  # None means sensor.max_fps

    # DVS configs
    dvs_fp_rate = fp_calibration.get('dvs_fp_rate', 2.0)

    for dvs in DVS_SENSORS:
        for theta in dvs_thetas:
            for coast in [False, True]:
                for seed in range(1, num_seeds + 1):
                    configs.append({
                        'sensor_name':     dvs.name,
                        'sensor_type':     'DVS',
                        'operating_point': f'theta={theta:.2f}',
                        'theta':           theta,
                        'fps_used':        None,
                        'coast':           coast,
                        'seed':            seed,
                        'power_mw':        round(dvs.p_static_mw, 4),
                        'cfg_kwargs': {
                            'resolution':           dvs.resolution,
                            'contrast_threshold':   theta,
                            'pixel_latency_s':      dvs.pixel_latency_us * 1e-6,
                            'refractory_cap':       dvs.max_event_rate_mevps * 1e6,
                            'coast':                coast,
                            'bg_fp_rate_per_frame': dvs_fp_rate,
                        },
                        'seq_dir':    MOT17_SEQ_DIR,
                        'max_frames': max_frames,
                        'here':       HERE,
                    })

    # CIS configs
    cis_fp_rate = fp_calibration.get('cis_fp_rate', 2.5)
    for cis in CIS_SENSORS:
        for fps in cis_fps_grid:
            actual_fps = cis.max_fps if fps is None else fps
            for seed in range(1, num_seeds + 1):
                configs.append({
                    'sensor_name':     cis.name,
                    'sensor_type':     'CIS',
                    'operating_point': f'fps={int(actual_fps)}',
                    'theta':           None,
                    'fps_used':        float(actual_fps),
                    'coast':           False,
                    'seed':            seed,
                    'power_mw':        round(cis.power_mw(actual_fps), 4),
                    'cfg_kwargs': {
                        'actual_fps':           float(actual_fps),
                        'resolution':           cis.resolution,
                        'adc_bits':             cis.adc_bits,
                        'bg_fp_rate_per_frame': cis_fp_rate,
                    },
                    'seq_dir':    MOT17_SEQ_DIR,
                    'max_frames': max_frames,
                    'here':       HERE,
                })

    return configs


def run_sweep_b(max_frames: int = 1050, num_seeds: int = 5,
                 num_workers: int | None = None,
                 out_csv: str | None = None,
                 fp_calibration: dict | None = None) -> pd.DataFrame:
    if num_workers is None:
        num_workers = min(28, os.cpu_count() or 4)
    if out_csv is None:
        out_csv = os.path.join(OUT_DIR, 'sweep_b_mot17.csv')
    if fp_calibration is None:
        fp_calibration = {}

    configs = _build_sweep_b_configs(max_frames, num_seeds, fp_calibration)
    num_configs = len(configs)
    print(f'[Sweep B] running {num_configs} configs on {num_workers} workers...')
    t0 = time.time()

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_run_one_sweep_b, c): c for c in configs}
        for future in as_completed(futures):
            done += 1
            try:
                rows.append(future.result())
            except Exception as exc:
                cfg = futures[future]
                print(f'  [{done}/{num_configs}] FAILED '
                      f"{cfg['sensor_name']} {cfg['operating_point']} "
                      f"seed={cfg['seed']}: {exc}")
            if done % 20 == 0 or done == num_configs:
                print(f'  [{done:3d}/{num_configs}] '
                      f'{time.time() - t0:.1f}s elapsed')

    wall = time.time() - t0
    df = pd.DataFrame(rows)
    if len(df) and 'sensor_type' in df.columns:
        df = df.sort_values(['sensor_type', 'sensor_name',
                              'operating_point', 'coast', 'seed'])
    df.to_csv(out_csv, index=False)
    num_failed = num_configs - len(df)
    print(f'[Sweep B] wrote {len(df)} rows to {out_csv} '
          f'({wall:.1f}s wall, {num_failed} failed)')
    if num_failed:
        print(f'[Sweep B] WARNING: {num_failed} configs failed; '
              f'rerun with --workers <smaller> or debug individual failures.')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-frames', type=int, default=1050)
    ap.add_argument('--num-seeds',  type=int, default=5)
    ap.add_argument('--workers',    type=int, default=None)
    ap.add_argument('--skip-a',     action='store_true')
    ap.add_argument('--skip-b',     action='store_true')
    args = ap.parse_args()

    if not args.skip_a:
        a_csv = os.path.join(OUT_DIR, 'sweep_a_analytical.csv')
        print(f'\n[Sweep A] writing to {a_csv}')
        run_sweep_a(a_csv)

    if not args.skip_b:
        print(f'\n[Sweep B] MOT17-04-SDP, '
              f'{args.max_frames} frames, {args.num_seeds} seeds')
        run_sweep_b(max_frames=args.max_frames, num_seeds=args.num_seeds,
                    num_workers=args.workers)


if __name__ == '__main__':
    main()
