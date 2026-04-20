"""SORT tracker https://arxiv.org/abs/1602.00763 (Bewley et al., 2016)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou_batch(dets_xyxy: np.ndarray, trks_xyxy: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes."""
    x1 = np.maximum(dets_xyxy[:, 0:1], trks_xyxy[:, 0:1].T)
    y1 = np.maximum(dets_xyxy[:, 1:2], trks_xyxy[:, 1:2].T)
    x2 = np.minimum(dets_xyxy[:, 2:3], trks_xyxy[:, 2:3].T)
    y2 = np.minimum(dets_xyxy[:, 3:4], trks_xyxy[:, 3:4].T)
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_d = (dets_xyxy[:, 2] - dets_xyxy[:, 0]) * (dets_xyxy[:, 3] - dets_xyxy[:, 1])
    area_t = (trks_xyxy[:, 2] - trks_xyxy[:, 0]) * (trks_xyxy[:, 3] - trks_xyxy[:, 1])
    return inter / (area_d[:, None] + area_t[None, :] - inter + 1e-9)


# Kalman constants. State is [cx, cy, s, r, vx, vy, vs] where s=area, r=aspect.
_F = np.eye(7, dtype=np.float64)
_F[0, 4] = _F[1, 5] = _F[2, 6] = 1.0  # position += velocity
_H = np.eye(4, 7, dtype=np.float64)  # observe [cx, cy, s, r]

_Q = np.eye(7, dtype=np.float64)
_Q[4:, 4:] *= 0.01
_Q[6, 6] *= 0.01

_R = np.eye(4, dtype=np.float64)
_R[2:, 2:] *= 10.0  # size measurements noisier


def _init_P() -> np.ndarray:
    P = np.eye(7, dtype=np.float64) * 10.0
    P[4:, 4:] *= 1000.0  # velocity starts uncertain
    return P


def _xyxy_to_z(bbox: np.ndarray) -> np.ndarray:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array([bbox[0] + w / 2, bbox[1] + h / 2, w * h, w / (h + 1e-9)])


def _z_to_xyxy(z: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(z[2] * z[3], 1e-9))
    h = z[2] / (w + 1e-9)
    return np.array([z[0] - w / 2, z[1] - h / 2, z[0] + w / 2, z[1] + h / 2])


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]]
    )


class _Track:
    _next_id = 1

    def __init__(self, bbox_xyxy: np.ndarray):
        self.id = _Track._next_id
        _Track._next_id += 1
        self.x = np.zeros(7, dtype=np.float64)
        self.x[:4] = _xyxy_to_z(bbox_xyxy)
        self.P = _init_P()
        self.time_since_update = 0
        self.hits = 1
        self.age = 0

    def predict(self) -> None:
        if self.x[6] + self.x[2] <= 0:  # area would go negative
            self.x[6] = 0.0
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + _Q
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_xyxy: np.ndarray) -> None:
        y = _xyxy_to_z(bbox_xyxy) - _H @ self.x  # innovation
        S = _H @ self.P @ _H.T + _R
        K = self.P @ _H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ _H) @ self.P
        self.time_since_update = 0
        self.hits += 1

    def bbox(self) -> np.ndarray:
        return _z_to_xyxy(self.x[:4])


class Sort:
    """SORT tracker: Kalman-predicted boxes matched via IoU + Hungarian."""

    def __init__(self, max_age: int = 5, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[_Track] = []
        self.frame_count = 0

    def update(self, detections_xywh) -> list[tuple[int, float, float, float, float]]:
        """One frame → confirmed tracks as (id, x, y, w, h)."""
        self.frame_count += 1

        # predict + drop NaN tracks
        for t in self.tracks:
            t.predict()
        self.tracks = [t for t in self.tracks if not np.any(np.isnan(t.bbox()))]

        dets = (
            _xywh_to_xyxy(np.asarray(detections_xywh, dtype=np.float64))
            if len(detections_xywh)
            else np.empty((0, 4))
        )

        matched, unmatched_dets = self._match(dets)
        for di, ti in matched:
            self.tracks[ti].update(dets[di])
        for di in unmatched_dets:
            self.tracks.append(_Track(dets[di]))

        # age out stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [self._report(t) for t in self.tracks if self._is_confirmed(t)]

    def _match(self, dets: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
        """Hungarian match gated by IoU threshold."""
        if len(dets) == 0 or len(self.tracks) == 0:
            return [], list(range(len(dets)))

        trk_bboxes = np.array([t.bbox() for t in self.tracks])
        iou = _iou_batch(dets, trk_bboxes)
        det_idx, trk_idx = linear_sum_assignment(1.0 - iou)

        matches: list[tuple[int, int]] = []
        matched_dets: set[int] = set()
        for di, ti in zip(det_idx, trk_idx):
            if iou[di, ti] >= self.iou_threshold:
                matches.append((int(di), int(ti)))
                matched_dets.add(int(di))
        unmatched = [i for i in range(len(dets)) if i not in matched_dets]
        return matches, unmatched

    def _is_confirmed(self, t: _Track) -> bool:
        # hot start: first few frames bypass the hit threshold
        return t.time_since_update == 0 and (
            t.hits >= self.min_hits or self.frame_count <= self.min_hits
        )

    @staticmethod
    def _report(t: _Track) -> tuple[int, float, float, float, float]:
        x1, y1, x2, y2 = t.bbox()
        return (t.id, float(x1), float(y1), float(x2 - x1), float(y2 - y1))
