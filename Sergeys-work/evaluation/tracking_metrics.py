"""HOTA / DetA / AssA / MOTA / IDF1 against MOT17 GT.

HOTA from https://arxiv.org/abs/2009.07736 (Luiten et al. 2021): per-alpha sqrt(DetA · AssA), averaged over
alpha in [0.05, 0.95]. MOTA/IDF1 straight from motmetrics at IoU 0.5.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

# motmetrics still calls np.asfarray, removed in newer numpy
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm


ALPHA_VALUES = np.linspace(0.05, 0.95, 19)


def _iou_matrix(gt_boxes: np.ndarray, pr_boxes: np.ndarray) -> np.ndarray:
    """IoU between two sets of (x, y, w, h) boxes."""
    if len(gt_boxes) == 0 or len(pr_boxes) == 0:
        return np.empty((len(gt_boxes), len(pr_boxes)))
    g = np.asarray(gt_boxes, dtype=float)
    p = np.asarray(pr_boxes, dtype=float)

    ix1 = np.maximum(g[:, 0:1], p[:, 0:1].T)
    iy1 = np.maximum(g[:, 1:2], p[:, 1:2].T)
    ix2 = np.minimum((g[:, 0] + g[:, 2])[:, None], (p[:, 0] + p[:, 2])[None, :])
    iy2 = np.minimum((g[:, 1] + g[:, 3])[:, None], (p[:, 1] + p[:, 3])[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

    union = (g[:, 2] * g[:, 3])[:, None] + (p[:, 2] * p[:, 3])[None, :] - inter
    return inter / (union + 1e-9)


def _dist_matrix(
    gt_boxes: np.ndarray, pr_boxes: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """1 - IoU, NaN below threshold (motmetrics format)."""
    iou = _iou_matrix(gt_boxes, pr_boxes)
    dist = 1.0 - iou
    dist[iou < iou_threshold] = np.nan
    return dist


def _frames_and_boxes(pred_df: pd.DataFrame, gt_df: pd.DataFrame):
    """Group pred/GT by frame → (frames, gt_by_frame, pr_by_frame)."""
    gt_by_frame = {int(f): g for f, g in gt_df.groupby("frame")}
    pr_by_frame = (
        {int(f): p for f, p in pred_df.groupby("frame")} if len(pred_df) else {}
    )
    frames = sorted(set(gt_by_frame) | set(pr_by_frame))
    return frames, gt_by_frame, pr_by_frame


def _accumulate(
    pred_df: pd.DataFrame, gt_df: pd.DataFrame, iou_threshold: float
) -> mm.MOTAccumulator:
    """Feed every frame into the motmetrics accumulator."""
    acc = mm.MOTAccumulator(auto_id=True)
    frames, gt_by_frame, pr_by_frame = _frames_and_boxes(pred_df, gt_df)
    for frame in frames:
        gt_frame = gt_by_frame.get(frame)
        pr_frame = pr_by_frame.get(frame)
        gt_ids = gt_frame["id"].tolist() if gt_frame is not None else []
        pr_ids = pr_frame["id"].tolist() if pr_frame is not None else []
        if not gt_ids and not pr_ids:
            continue
        gt_boxes = (
            gt_frame[["x", "y", "w", "h"]].to_numpy() if gt_ids else np.empty((0, 4))
        )
        pr_boxes = (
            pr_frame[["x", "y", "w", "h"]].to_numpy() if pr_ids else np.empty((0, 4))
        )
        acc.update(gt_ids, pr_ids, _dist_matrix(gt_boxes, pr_boxes, iou_threshold))
    return acc


# ── HOTA ────────────────────────────────────────────────────────────────


def compute_hota(
    pred_df: pd.DataFrame, gt_df: pd.DataFrame, alpha_values: np.ndarray | None = None
) -> dict:
    """HOTA/DetA/AssA averaged over alpha. Cols: frame, id, x, y, w, h."""
    if alpha_values is None:
        alpha_values = ALPHA_VALUES

    hotas, det_as, ass_as = [], [], []
    for alpha in alpha_values:
        acc = _accumulate(pred_df, gt_df, iou_threshold=(1.0 - alpha))
        det_a, ass_a = _det_ass_from_accumulator(acc)
        hotas.append(float(np.sqrt(max(det_a, 0.0) * max(ass_a, 0.0))))
        det_as.append(det_a)
        ass_as.append(ass_a)

    return {
        "hota": float(np.mean(hotas)),
        "det_a": float(np.mean(det_as)),
        "ass_a": float(np.mean(ass_as)),
        "hota_per_alpha": hotas,
        "det_a_per_alpha": det_as,
        "ass_a_per_alpha": ass_as,
        "alpha_values": list(map(float, alpha_values)),
    }


def _det_ass_from_accumulator(acc: mm.MOTAccumulator) -> tuple[float, float]:
    """DetA = TP/(TP+FN+FP). AssA = mean per-pair TPA/(TPA+FNA+FPA)."""
    events = acc.mot_events.reset_index()
    if len(events) == 0:
        return 0.0, 0.0

    matches = events[events["Type"].isin(["MATCH", "SWITCH"])]
    tp = len(matches)
    fn = (events["Type"] == "MISS").sum()
    fp = (events["Type"] == "FP").sum()

    det_a = tp / (tp + fn + fp) if (tp + fn + fp) else 0.0
    if tp == 0:
        return det_a, 0.0

    # per (gt, pred) pair: frames where they stayed matched
    pair_frames: dict[tuple, set] = {}
    gt_frames: dict[int, set] = {}
    pr_frames: dict[int, set] = {}
    for _, row in matches.iterrows():
        gid, pid = row["OId"], row["HId"]
        if pd.isna(gid) or pd.isna(pid):
            continue
        f = int(row["FrameId"])
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

    return det_a, float(np.mean(ratios))


# ── MOTA / IDF1 ─────────────────────────────────────────────────────────


def evaluate_mota(
    pred_df: pd.DataFrame, gt_df: pd.DataFrame, iou_threshold: float = 0.5
) -> dict:
    """MOTA/MOTP/IDF1 + switches + GT/pred counts."""
    acc = _accumulate(pred_df, gt_df, iou_threshold=(1.0 - iou_threshold))
    summary = mm.metrics.create().compute(
        acc,
        metrics=[
            "mota",
            "motp",
            "idf1",
            "num_switches",
            "num_objects",
            "num_predictions",
        ],
        name="seq",
    )
    row = summary.iloc[0]
    return {
        "mota": float(row["mota"]),
        "motp": float(row["motp"]) if not np.isnan(row["motp"]) else None,
        "idf1": float(row["idf1"]),
        "id_switches": int(row["num_switches"]),
        "num_gt": int(row["num_objects"]),
        "num_pred": int(row["num_predictions"]),
    }


def tracks_to_df(tracks_per_frame) -> pd.DataFrame:
    """[(frame, [(id, x, y, w, h), ...]), ...] → dataframe."""
    rows = [
        (frame, tid, x, y, w, h)
        for frame, tracks in tracks_per_frame
        for tid, x, y, w, h in tracks
    ]
    cols = ["frame", "id", "x", "y", "w", "h"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
