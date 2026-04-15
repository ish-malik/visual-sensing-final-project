"""HOTA, DetA and AssA for tracking evaluation.

Implements the metric from Luiten et al., IJCV 2021. For each IoU
threshold alpha between 0.05 and 0.95, the MOT accumulator gives us
match, miss and false positive events per frame. DetA is the standard
TP over TP plus FN plus FP. AssA walks the matched gt and pred id pairs
and counts how often the same pair stays matched versus how often each
side ends up matched to something else. HOTA at a given alpha is the
geometric mean of DetA and AssA, and the final score is the average
across all alphas.

Written to avoid pulling in trackeval just for three numbers.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd

if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm


ALPHA_VALUES = np.linspace(0.05, 0.95, 19)


def _iou_matrix(gt_boxes: np.ndarray, pr_boxes: np.ndarray) -> np.ndarray:
    """IoU matrix. gt_boxes and pr_boxes in (x, y, w, h)."""
    if len(gt_boxes) == 0 or len(pr_boxes) == 0:
        return np.empty((len(gt_boxes), len(pr_boxes)))
    g = np.asarray(gt_boxes, dtype=float)
    p = np.asarray(pr_boxes, dtype=float)

    gx2 = g[:, 0] + g[:, 2]
    gy2 = g[:, 1] + g[:, 3]
    px2 = p[:, 0] + p[:, 2]
    py2 = p[:, 1] + p[:, 3]

    ix1 = np.maximum(g[:, 0:1], p[:, 0:1].T)
    iy1 = np.maximum(g[:, 1:2], p[:, 1:2].T)
    ix2 = np.minimum(gx2[:, None], px2[None, :])
    iy2 = np.minimum(gy2[:, None], py2[None, :])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_g = g[:, 2] * g[:, 3]
    area_p = p[:, 2] * p[:, 3]
    union = area_g[:, None] + area_p[None, :] - inter + 1e-9
    return inter / union


def _dist_matrix_for_alpha(gt_boxes: np.ndarray, pr_boxes: np.ndarray,
                            alpha: float) -> np.ndarray:
    """Distance matrix with NaN where IoU < (1 - alpha)."""
    iou = _iou_matrix(gt_boxes, pr_boxes)
    dist = 1.0 - iou
    dist[iou < (1.0 - alpha)] = np.nan
    return dist


def hota_from_events(pred_df: pd.DataFrame, gt_df: pd.DataFrame,
                      alpha_values: np.ndarray | None = None) -> dict:
    """Compute HOTA, DetA, AssA from prediction and GT dataframes.

    Columns expected: frame, id, x, y, w, h for both DataFrames.
    """
    if alpha_values is None:
        alpha_values = ALPHA_VALUES

    frames = sorted(set(gt_df['frame'].unique()) |
                    set(pred_df['frame'].unique() if len(pred_df) else []))

    gt_by_frame = {int(f): g for f, g in gt_df.groupby('frame')}
    pr_by_frame = ({int(f): p for f, p in pred_df.groupby('frame')}
                   if len(pred_df) else {})

    hota_per_alpha = []
    det_a_per_alpha = []
    ass_a_per_alpha = []

    for alpha in alpha_values:
        acc = mm.MOTAccumulator(auto_id=True)

        for frame in frames:
            gt_frame = gt_by_frame.get(frame)
            pr_frame = pr_by_frame.get(frame)
            gt_ids = gt_frame['id'].tolist() if gt_frame is not None else []
            pr_ids = pr_frame['id'].tolist() if pr_frame is not None else []
            if not gt_ids and not pr_ids:
                continue
            gt_boxes = (gt_frame[['x', 'y', 'w', 'h']].to_numpy()
                        if gt_ids else np.empty((0, 4)))
            pr_boxes = (pr_frame[['x', 'y', 'w', 'h']].to_numpy()
                        if pr_ids else np.empty((0, 4)))
            dist = _dist_matrix_for_alpha(gt_boxes, pr_boxes, alpha)
            acc.update(gt_ids, pr_ids, dist)

        det_a, ass_a = _det_ass_from_accumulator(acc)
        hota_alpha = float(np.sqrt(max(det_a, 0.0) * max(ass_a, 0.0)))
        det_a_per_alpha.append(det_a)
        ass_a_per_alpha.append(ass_a)
        hota_per_alpha.append(hota_alpha)

    return {
        'hota':           float(np.mean(hota_per_alpha)),
        'det_a':          float(np.mean(det_a_per_alpha)),
        'ass_a':          float(np.mean(ass_a_per_alpha)),
        'hota_per_alpha': list(hota_per_alpha),
        'det_a_per_alpha': list(det_a_per_alpha),
        'ass_a_per_alpha': list(ass_a_per_alpha),
        'alpha_values':    list(map(float, alpha_values)),
    }


def _det_ass_from_accumulator(acc: mm.MOTAccumulator) -> tuple[float, float]:
    """Pull DetA and AssA out of a motmetrics event log.

    DetA is computed from the total match, miss and FP counts. AssA is
    computed by first collecting, for every matched gt and pred id, the
    set of frames where they were matched to each other. The association
    ratio for that pair is TPA over TPA plus FNA plus FPA, where FNA is
    frames where the gt was matched to a different pred and FPA is frames
    where the pred was matched to a different gt. AssA is the mean of
    those ratios across all pairs.
    """
    events = acc.mot_events.reset_index()
    if len(events) == 0:
        return 0.0, 0.0

    matches = events[events['Type'].isin(['MATCH', 'SWITCH'])]
    misses  = events[events['Type'] == 'MISS']
    fps     = events[events['Type'] == 'FP']

    tp = len(matches)
    fn = len(misses)
    fp = len(fps)

    if (tp + fn + fp) == 0:
        det_a = 0.0
    else:
        det_a = tp / (tp + fn + fp)

    if tp == 0:
        return det_a, 0.0

    # Build (gt_id, pred_id) -> set of frames where they were matched
    pair_frames: dict[tuple, set] = {}
    gt_frames: dict[int, set] = {}
    pr_frames: dict[int, set] = {}
    for _, row in matches.iterrows():
        gid = row['OId']
        pid = row['HId']
        f = int(row['FrameId'])
        if pd.isna(gid) or pd.isna(pid):
            continue
        pair_frames.setdefault((gid, pid), set()).add(f)
        gt_frames.setdefault(gid, set()).add(f)
        pr_frames.setdefault(pid, set()).add(f)

    if not pair_frames:
        return det_a, 0.0

    ratios = []
    for (gid, pid), frames_matched in pair_frames.items():
        tpa = len(frames_matched)
        fna = len(gt_frames[gid] - frames_matched)
        fpa = len(pr_frames[pid] - frames_matched)
        denom = tpa + fna + fpa
        ratios.append(tpa / denom if denom else 0.0)

    ass_a = float(np.mean(ratios)) if ratios else 0.0
    return det_a, ass_a


def compute_hota(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    """Top-level entry point used by the sweep driver."""
    return hota_from_events(pred_df, gt_df)
