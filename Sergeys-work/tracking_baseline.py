"""
Tracking Baseline: DVS vs CIS Position Error and Detection Completeness
Author: Sergey Petrushkevich
EECE5698 - Visual Sensing and Computing

Derives tracking quality metrics from sensor parameters to compare
DVS event-driven tracking vs CIS frame-based tracking.

Metrics computed:
  - Position error (pixels): how far the object moves between sensor updates
  - Missed fraction: what fraction of motion events/frames are lost
  - Tracking quality: combined score in [0, 1]
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)
MERGED_CSV = os.path.join(BASE, "merged_cis_dvs_data.csv")
DVS_TEMPORAL_CSV = os.path.join(PROJECT, "Ishs-work", "dvs_results",
                                 "dvs_temporal_variation.csv")
CIS_TEMPORAL_CSV = os.path.join(
    PROJECT, "Harshithas-work", "Spring-2026-ModuCIS-modeling-main",
    "ModuCIS.-CIS-modeling-main", "ModuCIS.-CIS-modeling-main",
    "CIS_Model", "Use_cases", "sweeps_results_final_cis_model",
    "cis_temporal_variation.csv")
OUT_DIR = BASE

# ── sensor constants ───────────────────────────────────────────────────
REFRACTORY_CAP = 18.75e6       # DVS max events/s (from dvs_model.py)
FALSE_POS_RATE = 0.10           # DVS false positive rate (from scene model)

# ── plot style (matches T4_comparison.py) ──────────────────────────────
CIS_COLOR = "#1f77b4"
DVS_COLOR = "#ff7f0e"
COLORS_SIZE = {25: "#1f77b4", 50: "#ff7f0e", 100: "#2ca02c"}
SIZE_MARKERS = {25: "o", 50: "s", 100: "D"}
BG_ORDER = ["low_texture", "high_texture"]
BG_LABELS = {
    "low_texture": "Low Texture (plain)",
    "high_texture": "High Texture (cluttered)",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ── helpers ────────────────────────────────────────────────────────────

def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def add_finding(ax, text, loc="lower right", fontsize=9):
    box_props = dict(
        boxstyle="round,pad=0.5",
        facecolor="lightyellow",
        edgecolor="#ccaa44",
        alpha=0.92,
        linewidth=1.2,
    )
    anchors = {
        "lower right": (0.97, 0.03, "right", "bottom"),
        "lower left": (0.03, 0.03, "left", "bottom"),
        "upper right": (0.97, 0.97, "right", "top"),
        "upper left": (0.03, 0.97, "left", "top"),
        "center right": (0.97, 0.50, "right", "center"),
        "lower center": (0.50, 0.03, "center", "bottom"),
    }
    x, y, ha, va = anchors.get(loc, anchors["lower right"])
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment=va, horizontalalignment=ha, bbox=box_props)


# ── core tracking computations ─────────────────────────────────────────

def compute_dvs_tracking(row):
    """Derive DVS tracking metrics from a merged-data row."""
    event_rate_eff = row["event_rate_eff"]
    event_rate_scaled = row["event_rate_scaled"]
    velocity = row["velocity_px_s"]

    # position error: object displacement between consecutive events
    if event_rate_eff > 0:
        inter_event_time = 1.0 / event_rate_eff
    else:
        inter_event_time = np.inf
    position_error_px = velocity * inter_event_time

    # missed fraction: events lost to refractory saturation
    if event_rate_scaled > REFRACTORY_CAP:
        missed_fraction = 1.0 - (REFRACTORY_CAP / event_rate_scaled)
    else:
        missed_fraction = 0.0

    # detection completeness: penalized by noise (false positives)
    detection_completeness = (1.0 - missed_fraction) / (1.0 + FALSE_POS_RATE)

    # tracking quality: combined metric in [0, 1]
    tracking_quality = detection_completeness / (1.0 + position_error_px)

    return {
        "dvs_position_error_px": round(position_error_px, 6),
        "dvs_inter_event_time_s": round(inter_event_time, 9),
        "dvs_missed_fraction": round(missed_fraction, 6),
        "dvs_detection_completeness": round(detection_completeness, 6),
        "dvs_tracking_quality": round(tracking_quality, 6),
    }


def compute_cis_tracking(row):
    """Derive CIS tracking metrics from a merged-data row."""
    velocity = row["velocity_px_s"]
    obj_size = row["object_size_px"]
    required_fps = row["required_fps"]
    max_fps = row["max_frame_rate_hz"]

    # actual fps (clipped to hardware limit)
    actual_fps = min(required_fps, max_fps)

    # position error: inter-frame displacement + motion blur
    frame_time = 1.0 / actual_fps
    inter_frame_disp = velocity * frame_time
    motion_blur_px = 0.5 * velocity * frame_time
    position_error_px = inter_frame_disp + motion_blur_px  # = 1.5 * v / fps

    # missed fraction: temporal under-sampling when fps is clipped
    if required_fps > max_fps:
        missed_fraction = 1.0 - (max_fps / required_fps)
    else:
        missed_fraction = 0.0

    # detection completeness: degraded by missed frames and motion blur
    blur_quality = obj_size / (obj_size + motion_blur_px)
    detection_completeness = (1.0 - missed_fraction) * blur_quality

    # tracking quality: combined metric in [0, 1]
    tracking_quality = detection_completeness / (1.0 + position_error_px)

    return {
        "cis_actual_fps": round(actual_fps, 2),
        "cis_frame_time_s": round(frame_time, 6),
        "cis_position_error_px": round(position_error_px, 4),
        "cis_motion_blur_px": round(motion_blur_px, 4),
        "cis_missed_fraction": round(missed_fraction, 6),
        "cis_detection_completeness": round(detection_completeness, 6),
        "cis_tracking_quality": round(tracking_quality, 6),
    }


def compute_all_tracking(df):
    """Add DVS and CIS tracking columns to the merged DataFrame."""
    dvs_cols = df.apply(compute_dvs_tracking, axis=1, result_type="expand")
    cis_cols = df.apply(compute_cis_tracking, axis=1, result_type="expand")
    return pd.concat([df, dvs_cols, cis_cols], axis=1)


def save_results(df):
    cols = [
        "object_size_px", "velocity_px_s", "background",
        "threshold_name", "theta",
        "event_rate_eff", "event_rate_scaled",
        "dvs_position_error_px", "dvs_inter_event_time_s",
        "dvs_missed_fraction", "dvs_detection_completeness", "dvs_tracking_quality",
        "cis_actual_fps", "cis_frame_time_s",
        "cis_position_error_px", "cis_motion_blur_px",
        "cis_missed_fraction", "cis_detection_completeness", "cis_tracking_quality",
    ]
    out_path = os.path.join(OUT_DIR, "tracking_baseline_results.csv")
    df[cols].to_csv(out_path, index=False)
    print(f"  Saved tracking_baseline_results.csv ({len(df)} rows)")


# ── plots ──────────────────────────────────────────────────────────────

def plot_t1_position_error_vs_velocity(df):
    """Position error vs velocity: DVS vs CIS, split by background."""
    sub = df[df["threshold_name"] == "med_threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, bg in zip(axes, BG_ORDER):
        bg_df = sub[sub["background"] == bg]

        # shade the CIS fps-clipped region
        clipped_vels = bg_df[bg_df["fps_clipped_warning"] == 1]["velocity_px_s"]
        if not clipped_vels.empty:
            clip_start = clipped_vels.min()
            ax.axvspan(clip_start, bg_df["velocity_px_s"].max() * 1.02,
                        alpha=0.08, color="#d62728", zorder=0)
            ax.axvline(x=clip_start, color="#d62728", linestyle=":",
                        alpha=0.5, linewidth=1.5)
            ax.text(clip_start, 0.98, f" CIS fps limit",
                    transform=ax.get_xaxis_transform(),
                    fontsize=8, color="#d62728", va="top")

        for sz in sorted(bg_df["object_size_px"].unique()):
            sz_df = bg_df[bg_df["object_size_px"] == sz].sort_values("velocity_px_s")
            color = COLORS_SIZE[sz]
            marker = SIZE_MARKERS[sz]
            ax.plot(sz_df["velocity_px_s"], sz_df["dvs_position_error_px"],
                    marker=marker, color=color, linestyle="--", linewidth=1.5,
                    label=f"DVS {sz}px")

            # CIS: solid for feasible, faded for clipped
            ok = sz_df[sz_df["fps_clipped_warning"] == 0]
            clip = sz_df[sz_df["fps_clipped_warning"] == 1]
            if not ok.empty:
                ax.plot(ok["velocity_px_s"], ok["cis_position_error_px"],
                        marker=marker, color=color, linestyle="-", linewidth=2,
                        label=f"CIS {sz}px")
            if not clip.empty:
                ax.plot(clip["velocity_px_s"], clip["cis_position_error_px"],
                        marker=marker, color=color, linestyle="-", linewidth=1.2,
                        alpha=0.35)
            if not ok.empty and not clip.empty:
                bridge = pd.concat([ok.iloc[[-1]], clip.iloc[[0]]])
                ax.plot(bridge["velocity_px_s"], bridge["cis_position_error_px"],
                        linestyle="-", linewidth=1.2, color=color, alpha=0.35)

        ax.set_yscale("log")
        ax.set_xlabel("Velocity (px/s)")
        ax.set_title(BG_LABELS[bg])
        ax.legend(fontsize=8, ncol=2)

    axes[0].set_ylabel("Position Error (pixels, log scale)")
    fig.suptitle("Tracking Position Error: DVS vs CIS\n"
                 "(dashed = DVS, solid = CIS, shaded = CIS fps limit exceeded)",
                 fontweight="bold")
    add_finding(axes[1],
                "DVS: sub-pixel error at all velocities\n"
                "CIS: error grows with velocity\n"
                "Shaded zone: CIS fps clipped",
                loc="upper left")
    plt.tight_layout()
    save(fig, "tracking_position_error_vs_velocity.png")


def plot_t2_missed_fraction(df):
    """Missed fraction vs velocity: grouped bars for CIS, DVS always zero."""
    sub = df[(df["threshold_name"] == "med_threshold") &
             (df["object_size_px"] == 50)].sort_values("velocity_px_s")

    fig, ax = plt.subplots(figsize=(10, 6))
    velocities = sorted(sub["velocity_px_s"].unique())
    x = np.arange(len(velocities))
    width = 0.25

    for i, bg in enumerate(BG_ORDER):
        bg_df = sub[sub["background"] == bg].sort_values("velocity_px_s")
        ax.bar(x + i * width, bg_df["cis_missed_fraction"].values,
               width, label=f"CIS — {BG_LABELS[bg]}", alpha=0.85,
               color=CIS_COLOR if i == 0 else "#6baed6")

    # DVS missed fraction (always 0) shown as thin line
    ax.axhline(y=0, color=DVS_COLOR, linewidth=2, linestyle="--",
               label="DVS (zero missed events)")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([str(v) for v in velocities])
    ax.set_xlabel("Velocity (px/s)")
    ax.set_ylabel("Missed Fraction")
    ax.set_title("Missed Motion Fraction: CIS vs DVS (object = 50px)")
    ax.legend()
    ax.set_ylim(-0.05, 1.0)

    add_finding(ax,
                f"DVS: zero missed events (max {REFRACTORY_CAP/1e6:.1f}M ev/s capacity)\n"
                "CIS: misses motion when fps clipped by hardware limit",
                loc="upper left")
    plt.tight_layout()
    save(fig, "tracking_missed_fraction.png")


def plot_t3_tracking_quality(df):
    """Tracking quality [0,1] vs velocity: DVS vs CIS, split by background."""
    sub = df[df["threshold_name"] == "med_threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, bg in zip(axes, BG_ORDER):
        bg_df = sub[sub["background"] == bg]
        for sz in sorted(bg_df["object_size_px"].unique()):
            sz_df = bg_df[bg_df["object_size_px"] == sz].sort_values("velocity_px_s")
            color = COLORS_SIZE[sz]
            marker = SIZE_MARKERS[sz]
            ax.plot(sz_df["velocity_px_s"], sz_df["dvs_tracking_quality"],
                    marker=marker, color=color, linestyle="--", linewidth=1.5,
                    label=f"DVS {sz}px")
            ax.plot(sz_df["velocity_px_s"], sz_df["cis_tracking_quality"],
                    marker=marker, color=color, linestyle="-", linewidth=2,
                    label=f"CIS {sz}px")

        ax.axhline(y=0.5, color="gray", linewidth=1, linestyle=":",
                    alpha=0.7, label="Quality = 0.5 (degraded)")
        ax.set_xlabel("Velocity (px/s)")
        ax.set_title(BG_LABELS[bg])
        ax.legend(fontsize=8, ncol=2)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Tracking Quality [0 = lost, 1 = perfect]")
    fig.suptitle("Tracking Quality: DVS vs CIS\n"
                 "(dashed = DVS, solid = CIS)",
                 fontweight="bold")
    add_finding(axes[1],
                "DVS quality stays near 0.91\n"
                "CIS degrades sharply with velocity",
                loc="center right")
    plt.tight_layout()
    save(fig, "tracking_quality_vs_velocity.png")


def plot_t4_tracking_heatmap(df):
    """2×2 heatmap: rows = sensor (DVS, CIS), cols = background."""
    sub = df[df["threshold_name"] == "med_threshold"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col_idx, bg in enumerate(BG_ORDER):
        bg_df = sub[sub["background"] == bg]

        # DVS heatmap (top row)
        dvs_pivot = bg_df.pivot_table(
            index="velocity_px_s", columns="object_size_px",
            values="dvs_tracking_quality", aggfunc="first"
        ).sort_index(ascending=False)
        im = axes[0, col_idx].imshow(
            dvs_pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto"
        )
        _annotate_heatmap(axes[0, col_idx], dvs_pivot)
        axes[0, col_idx].set_title(f"DVS — {BG_LABELS[bg]}")

        # CIS heatmap (bottom row)
        cis_pivot = bg_df.pivot_table(
            index="velocity_px_s", columns="object_size_px",
            values="cis_tracking_quality", aggfunc="first"
        ).sort_index(ascending=False)
        axes[1, col_idx].imshow(
            cis_pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto"
        )
        _annotate_heatmap(axes[1, col_idx], cis_pivot)
        axes[1, col_idx].set_title(f"CIS — {BG_LABELS[bg]}")

        # hatched overlay on infeasible CIS cells
        from matplotlib.patches import Rectangle
        for row_i, vel in enumerate(cis_pivot.index):
            for col_j, sz in enumerate(cis_pivot.columns):
                row_data = bg_df[(bg_df["velocity_px_s"] == vel) &
                                 (bg_df["object_size_px"] == sz)]
                if not row_data.empty and row_data["fps_clipped_warning"].values[0] == 1:
                    rect = Rectangle((col_j - 0.5, row_i - 0.5), 1, 1,
                                     fill=False, hatch="///", edgecolor="white",
                                     linewidth=1.2, zorder=5)
                    axes[1, col_idx].add_patch(rect)

    # axis labels
    for row_idx in range(2):
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            bg = BG_ORDER[col_idx]
            bg_df = sub[sub["background"] == bg]
            pivot = bg_df.pivot_table(
                index="velocity_px_s", columns="object_size_px",
                values="dvs_tracking_quality", aggfunc="first"
            ).sort_index(ascending=False)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(v) for v in pivot.index])
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(s) for s in pivot.columns])
            ax.set_ylabel("Velocity (px/s)")
            ax.set_xlabel("Object Size (px)")

    fig.colorbar(im, ax=axes, label="Tracking Quality [0-1]",
                 shrink=0.6, pad=0.02)
    fig.suptitle("Tracking Quality Heatmap: DVS (top) vs CIS (bottom)\n"
                 "Hatched cells = CIS fps clipped by hardware limit",
                 fontweight="bold")
    save(fig, "tracking_quality_heatmap.png")


def _annotate_heatmap(ax, pivot):
    """Write cell values on a heatmap."""
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)


def plot_t5_error_vs_sensor_params(df):
    """Left: DVS error vs event rate. Right: CIS error vs actual fps."""
    sub = df[df["threshold_name"] == "med_threshold"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ── left: DVS position error vs event rate ──
    for sz in sorted(sub["object_size_px"].unique()):
        sz_df = sub[sub["object_size_px"] == sz]
        sc = ax1.scatter(sz_df["event_rate_eff"], sz_df["dvs_position_error_px"],
                         c=sz_df["velocity_px_s"], cmap="viridis",
                         marker=SIZE_MARKERS[sz], s=80, edgecolors="k",
                         linewidths=0.5, label=f"{sz}px", zorder=3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("DVS Event Rate (events/s)")
    ax1.set_ylabel("DVS Position Error (px)")
    ax1.set_title("DVS: Position Error vs Event Rate")
    ax1.legend(title="Object size")
    fig.colorbar(sc, ax=ax1, label="Velocity (px/s)", shrink=0.8)

    # theoretical reference line: error = v / event_rate for v=500
    er_range = np.logspace(np.log10(sub["event_rate_eff"].min()),
                           np.log10(sub["event_rate_eff"].max()), 50)
    ax1.plot(er_range, 500 / er_range, "k--", alpha=0.4,
             label="theoretical (v=500)")
    ax1.legend(title="Object size", fontsize=8)

    add_finding(ax1,
                "More events → lower error\n"
                "All DVS errors are sub-pixel",
                loc="upper right")

    # ── right: CIS position error vs actual fps (the real bottleneck) ──
    for bg in BG_ORDER:
        bg_df = sub[sub["background"] == bg]
        for sz in sorted(bg_df["object_size_px"].unique()):
            sz_df = bg_df[bg_df["object_size_px"] == sz].sort_values("velocity_px_s")
            marker = SIZE_MARKERS[sz]
            label = f"{BG_LABELS[bg][:4]} {sz}px"
            sc2 = ax2.scatter(sz_df["cis_actual_fps"], sz_df["cis_position_error_px"],
                        c=sz_df["velocity_px_s"], cmap="viridis",
                        marker=marker, s=80, edgecolors="k", linewidths=0.5,
                        label=label, zorder=3)
            # faded markers for clipped points
            clipped = sz_df[sz_df["fps_clipped_warning"] == 1]
            if not clipped.empty:
                ax2.scatter(clipped["cis_actual_fps"], clipped["cis_position_error_px"],
                           c=clipped["velocity_px_s"], cmap="viridis",
                           marker=marker, s=80, edgecolors="k", linewidths=0.5,
                           alpha=0.3, zorder=2)

    # vertical line at CIS hardware limit
    max_fps = sub["max_frame_rate_hz"].iloc[0]
    ax2.axvline(x=max_fps, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.text(max_fps - 3, ax2.get_ylim()[1] * 0.9 if ax2.get_ylim()[1] > 0 else 30,
             f"Hardware limit\n({max_fps:.0f} fps)",
             fontsize=8, color="#d62728", ha="right", va="top")

    ax2.set_xlabel("CIS Actual Frame Rate (fps)")
    ax2.set_ylabel("CIS Position Error (px)")
    ax2.set_title("CIS: Position Error vs Frame Rate")
    ax2.legend(fontsize=7, ncol=2, title="Background / Size")
    fig.colorbar(sc2, ax=ax2, label="Velocity (px/s)", shrink=0.8)

    add_finding(ax2,
                "All CIS points cluster at max 178 fps\n"
                "Faded = fps clipped by hardware\n"
                "Position error grows with velocity",
                loc="upper right")

    fig.suptitle("Tracking Error vs Sensor Parameters\n"
                 "DVS: event rate drives error | CIS: frame rate limits tracking",
                 fontweight="bold")
    plt.tight_layout()
    save(fig, "tracking_error_vs_sensor_params.png")


def plot_t6_slide_summary(df):
    """Compact 2×2 summary figure for the TRACKING MODEL AND ANALYSIS slide."""
    sub = df[df["threshold_name"] == "med_threshold"]

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # ── top-left: position error vs velocity (core result) ──
    ax1 = fig.add_subplot(gs[0, 0])
    for sz in sorted(sub["object_size_px"].unique()):
        sz_df = sub[(sub["object_size_px"] == sz) &
                     (sub["background"] == "low_texture")].sort_values("velocity_px_s")
        color = COLORS_SIZE[sz]
        marker = SIZE_MARKERS[sz]
        ax1.plot(sz_df["velocity_px_s"], sz_df["dvs_position_error_px"],
                marker=marker, color=color, linestyle="--", linewidth=1.5,
                label=f"DVS {sz}px")
        # split CIS into feasible (solid) and clipped (faded)
        ok = sz_df[sz_df["fps_clipped_warning"] == 0]
        clip = sz_df[sz_df["fps_clipped_warning"] == 1]
        if not ok.empty:
            ax1.plot(ok["velocity_px_s"], ok["cis_position_error_px"],
                    marker=marker, color=color, linestyle="-", linewidth=2,
                    label=f"CIS {sz}px")
        if not clip.empty:
            ax1.plot(clip["velocity_px_s"], clip["cis_position_error_px"],
                    marker=marker, color=color, linestyle="-", linewidth=1.2,
                    alpha=0.35)
        if not ok.empty and not clip.empty:
            bridge = pd.concat([ok.iloc[[-1]], clip.iloc[[0]]])
            ax1.plot(bridge["velocity_px_s"], bridge["cis_position_error_px"],
                    linestyle="-", linewidth=1.2, color=color, alpha=0.35)

    # shade the CIS limit zone
    clipped_vels = sub[(sub["background"] == "low_texture") &
                        (sub["fps_clipped_warning"] == 1)]["velocity_px_s"]
    if not clipped_vels.empty:
        ax1.axvspan(clipped_vels.min(), sub["velocity_px_s"].max() * 1.02,
                    alpha=0.08, color="#d62728", zorder=0)

    ax1.set_yscale("log")
    ax1.set_xlabel("Velocity (px/s)")
    ax1.set_ylabel("Position Error (px, log)")
    ax1.set_title("Position Error: DVS vs CIS", fontweight="bold")
    ax1.legend(fontsize=7, ncol=2)
    add_finding(ax1, "DVS: <0.004 px (sub-pixel)\nCIS: up to 33 px at 2000 px/s\nShaded = CIS fps clipped",
                loc="upper left", fontsize=8)

    # ── top-right: tracking quality heatmap (DVS vs CIS side-by-side) ──
    ax2a = fig.add_subplot(gs[0, 1])
    # show just low-texture: DVS top half, CIS bottom half
    bg_df = sub[sub["background"] == "low_texture"]

    dvs_pivot = bg_df.pivot_table(
        index="velocity_px_s", columns="object_size_px",
        values="dvs_tracking_quality", aggfunc="first"
    ).sort_index(ascending=False)
    cis_pivot = bg_df.pivot_table(
        index="velocity_px_s", columns="object_size_px",
        values="cis_tracking_quality", aggfunc="first"
    ).sort_index(ascending=False)

    # stack DVS on top of CIS
    combined = np.vstack([dvs_pivot.values, cis_pivot.values])
    im = ax2a.imshow(combined, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    n_vel = len(dvs_pivot.index)
    n_sz = len(dvs_pivot.columns)
    for i in range(2 * n_vel):
        for j in range(n_sz):
            val = combined[i, j]
            color = "white" if val < 0.4 else "black"
            ax2a.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    ax2a.axhline(y=n_vel - 0.5, color="white", linewidth=3)
    vel_labels = [str(v) for v in dvs_pivot.index]
    ax2a.set_yticks(range(2 * n_vel))
    ax2a.set_yticklabels(vel_labels + vel_labels, fontsize=8)
    ax2a.set_xticks(range(n_sz))
    ax2a.set_xticklabels([str(s) for s in dvs_pivot.columns])
    ax2a.set_xlabel("Object Size (px)")
    ax2a.set_ylabel("Velocity (px/s)")
    ax2a.set_title("Tracking Quality [0-1]\nDVS (top) vs CIS (bottom)", fontweight="bold")
    fig.colorbar(im, ax=ax2a, shrink=0.7, label="Quality")

    # ── bottom-left: DVS error vs event rate ──
    ax3 = fig.add_subplot(gs[1, 0])
    for sz in sorted(sub["object_size_px"].unique()):
        sz_df = sub[sub["object_size_px"] == sz]
        sc = ax3.scatter(sz_df["event_rate_eff"], sz_df["dvs_position_error_px"],
                         c=sz_df["velocity_px_s"], cmap="viridis",
                         marker=SIZE_MARKERS[sz], s=70, edgecolors="k",
                         linewidths=0.4, label=f"{sz}px", zorder=3)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("DVS Event Rate (events/s)")
    ax3.set_ylabel("Position Error (px)")
    ax3.set_title("DVS: Error vs Event Rate", fontweight="bold")
    ax3.legend(title="Obj size", fontsize=7)
    fig.colorbar(sc, ax=ax3, label="Velocity (px/s)", shrink=0.7)
    add_finding(ax3, "Higher event rate → lower error\nerror = velocity / event_rate",
                loc="upper right", fontsize=8)

    # ── bottom-right: missed fraction bar chart ──
    ax4 = fig.add_subplot(gs[1, 1])
    bar_sub = sub[sub["object_size_px"] == 50].sort_values("velocity_px_s")
    velocities = sorted(bar_sub["velocity_px_s"].unique())
    x = np.arange(len(velocities))
    width = 0.3
    for i, bg in enumerate(BG_ORDER):
        bg_vals = bar_sub[bar_sub["background"] == bg].sort_values("velocity_px_s")
        ax4.bar(x + i * width, bg_vals["cis_missed_fraction"].values,
               width, label=f"CIS — {BG_LABELS[bg]}", alpha=0.85,
               color=CIS_COLOR if i == 0 else "#6baed6")
    ax4.axhline(y=0, color=DVS_COLOR, linewidth=2, linestyle="--",
               label="DVS (zero missed)")
    ax4.set_xticks(x + width / 2)
    ax4.set_xticklabels([str(v) for v in velocities], fontsize=8)
    ax4.set_xlabel("Velocity (px/s)")
    ax4.set_ylabel("Missed Fraction")
    ax4.set_title("Missed Motion Events", fontweight="bold")
    ax4.legend(fontsize=7)
    ax4.set_ylim(-0.05, 1.0)
    add_finding(ax4, "CIS misses 80%+ at 2000 px/s\nDVS: zero missed events",
                loc="upper left", fontsize=8)

    fig.suptitle("TRACKING MODEL AND ANALYSIS\n"
                 "DVS Event-Driven Tracking vs CIS Frame-Based Tracking",
                 fontsize=16, fontweight="bold", y=0.98)
    save(fig, "tracking_slide_summary.png")


def plot_t9_power_tracking_decision(df):
    """The 'which sensor?' figure. Ties together active_fraction, power,
    tracking quality, and dvs_can_track into one unified comparison."""
    med = df[df["threshold_name"] == "med_threshold"]

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # ── (A) Tracking efficiency: quality per milliwatt ──
    ax_a = fig.add_subplot(gs[0, 0])
    for bg in BG_ORDER:
        bg_df = med[(med["background"] == bg) &
                     (med["object_size_px"] == 50)].sort_values("velocity_px_s")
        ls = "-" if bg == "low_texture" else "--"
        dvs_eff = bg_df["dvs_tracking_quality"] / bg_df["dvs_power_mW"] * 1000
        cis_eff = bg_df["cis_tracking_quality"] / bg_df["cis_power_mW"] * 1000
        bg_short = "low tex" if bg == "low_texture" else "high tex"
        ax_a.plot(bg_df["velocity_px_s"], dvs_eff, marker="o", linestyle=ls,
                  color=DVS_COLOR, linewidth=2, label=f"DVS ({bg_short})")
        ax_a.plot(bg_df["velocity_px_s"], cis_eff, marker="s", linestyle=ls,
                  color=CIS_COLOR, linewidth=2, label=f"CIS ({bg_short})")
    ax_a.set_xlabel("Velocity (px/s)")
    ax_a.set_ylabel("Tracking Quality per Watt")
    ax_a.set_title("A. Tracking Efficiency\n(higher = more tracking per power dollar)",
                     fontweight="bold")
    ax_a.legend(fontsize=8, loc="center right")
    ax_a.set_yscale("log")

    # ── (B) Why: pixel utilization drives the power gap ──
    ax_b = fig.add_subplot(gs[0, 1])
    lt50 = med[(med["background"] == "low_texture") &
                (med["object_size_px"] == 50)].sort_values("velocity_px_s")
    n_total = lt50["n_total_pixels"].iloc[0]
    vel = lt50["velocity_px_s"].values

    # DVS reads only active pixels; CIS reads all pixels * fps
    ax_b.fill_between(vel, 0, lt50["active_fraction"].values * 100,
                       alpha=0.4, color=DVS_COLOR, label="DVS active pixels")
    ax_b.fill_between(vel, lt50["active_fraction"].values * 100, 100,
                       alpha=0.15, color="gray", label="DVS silent pixels (no power cost)")
    ax_b.axhline(y=100, color=CIS_COLOR, linewidth=3,
                  label="CIS: reads 100% every frame")

    for i, v in enumerate(vel):
        n_act = lt50["n_active_pixels"].values[i]
        pct = lt50["active_fraction"].values[i] * 100
        if pct > 0.5:
            ax_b.text(v, pct + 3, f"{n_act:,.0f}\n({pct:.1f}%)",
                      ha="center", fontsize=7, color=DVS_COLOR, fontweight="bold")

    ax_b.set_xlabel("Velocity (px/s)")
    ax_b.set_ylabel("Pixel Utilization (%)")
    ax_b.set_title(f"B. Why DVS Wins: Pixel Efficiency\n({n_total:,} total pixels)",
                     fontweight="bold")
    ax_b.legend(fontsize=8, loc="center right")
    ax_b.set_ylim(-2, 115)

    # ── (C) Power per active pixel ──
    ax_c = fig.add_subplot(gs[0, 2])
    dvs_power_per_active = lt50["dvs_power_mW"].values / lt50["n_active_pixels"].values
    # CIS "active pixels" = all pixels (it reads everything)
    cis_power_per_pixel = lt50["cis_power_mW"].values / n_total
    ax_c.bar(np.arange(len(vel)) - 0.18, dvs_power_per_active * 1e3, 0.36,
             label="DVS (per active pixel)", color=DVS_COLOR, alpha=0.85)
    ax_c.bar(np.arange(len(vel)) + 0.18, cis_power_per_pixel * 1e3, 0.36,
             label="CIS (per pixel)", color=CIS_COLOR, alpha=0.85)
    ax_c.set_xticks(np.arange(len(vel)))
    ax_c.set_xticklabels([str(v) for v in vel], fontsize=8)
    ax_c.set_xlabel("Velocity (px/s)")
    ax_c.set_ylabel("Power per Pixel (μW)")
    ax_c.set_title("C. Power Cost per Pixel\n(DVS charges only active pixels)",
                     fontweight="bold")
    ax_c.legend(fontsize=8, loc="upper left")

    # ── (D) Feasibility: can each sensor track? ──
    ax_d = fig.add_subplot(gs[1, 0])
    # DVS: dvs_can_track flag for all conditions
    dvs_ok = med.groupby("velocity_px_s")["dvs_can_track"].all()
    cis_ok = med.groupby("velocity_px_s").apply(
        lambda g: (g["fps_clipped_warning"] == 0).all())
    vels = sorted(med["velocity_px_s"].unique())
    x = np.arange(len(vels))
    colors_dvs = ["#2ca02c" if dvs_ok.get(v, False) else "#d62728" for v in vels]
    colors_cis = ["#2ca02c" if cis_ok.get(v, False) else "#d62728" for v in vels]
    ax_d.bar(x - 0.2, [1]*len(vels), 0.38, color=colors_dvs, alpha=0.8,
             edgecolor="white", linewidth=0.5)
    ax_d.bar(x + 0.2, [1]*len(vels), 0.38, color=colors_cis, alpha=0.8,
             edgecolor="white", linewidth=0.5)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([str(v) for v in vels], fontsize=8)
    ax_d.set_xlabel("Velocity (px/s)")
    ax_d.set_yticks([])
    ax_d.set_title("D. Can the Sensor Track?\n(green = yes for ALL sizes/backgrounds)",
                     fontweight="bold")
    # custom legend
    from matplotlib.patches import Patch
    ax_d.legend(handles=[
        Patch(facecolor="#2ca02c", label="Can track"),
        Patch(facecolor="#d62728", label="Cannot track (some scenes)"),
    ], fontsize=9, loc="upper right")
    ax_d.text(0, 0.5, "DVS", ha="center", va="center", fontsize=10,
              fontweight="bold", color="white")
    ax_d.text(0.38, 0.5, "CIS", ha="center", va="center", fontsize=10,
              fontweight="bold", color="white")
    # label bars
    for i in range(len(vels)):
        ax_d.text(i - 0.2, 0.5, "DVS", ha="center", va="center",
                  fontsize=7, fontweight="bold", color="white")
        ax_d.text(i + 0.2, 0.5, "CIS", ha="center", va="center",
                  fontsize=7, fontweight="bold", color="white")

    # ── (E) Summary: power savings × tracking quality advantage ──
    ax_e = fig.add_subplot(gs[1, 1])
    for sz in sorted(med["object_size_px"].unique()):
        sz_df = med[(med["object_size_px"] == sz) &
                     (med["background"] == "low_texture")].sort_values("velocity_px_s")
        quality_advantage = sz_df["dvs_tracking_quality"] / sz_df["cis_tracking_quality"]
        power_savings = sz_df["cis_power_mW"] / sz_df["dvs_power_mW"]
        ax_e.scatter(power_savings, quality_advantage,
                     marker=SIZE_MARKERS[sz], s=100, color=COLORS_SIZE[sz],
                     edgecolors="k", linewidths=0.5, label=f"{sz}px", zorder=3)
    ax_e.set_xlabel("Power Savings (CIS/DVS ratio, higher = DVS cheaper)")
    ax_e.set_ylabel("Tracking Quality Advantage (DVS/CIS ratio)")
    ax_e.set_title("E. DVS Advantage: Power Savings × Tracking Quality",
                     fontweight="bold")
    ax_e.legend(title="Object size", fontsize=9)
    ax_e.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
    ax_e.axvline(x=1, color="gray", linestyle=":", alpha=0.5)
    ax_e.text(15, 2, "DVS is better\non BOTH axes", fontsize=9, color="gray",
              style="italic", ha="center")

    # ── (F) Design rule text ──
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")

    n_total_scenes = len(med[["object_size_px","velocity_px_s","background"]].drop_duplicates())
    n_cis_fail = len(med[med["fps_clipped_warning"]==1][
        ["object_size_px","velocity_px_s","background"]].drop_duplicates())
    min_ratio = med["power_ratio"].min()
    max_ratio = med["power_ratio"].max()

    rule_text = (
        "DESIGN DECISION\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "USE DVS when:\n"
        f"  • Power efficiency matters\n"
        f"    ({min_ratio:.0f}–{max_ratio:.0f}x less power)\n"
        f"  • Object tracking is the goal\n"
        f"    (0.91 quality vs 0.002–0.20)\n"
        f"  • High velocity scenes (>500 px/s)\n"
        f"    DVS: 0/{n_total_scenes} failures\n"
        f"    CIS: {n_cis_fail}/{n_total_scenes} fps-clipped\n\n"
        "USE CIS when:\n"
        "  • Full-frame RGB image needed\n"
        "  • Object recognition/classification\n"
        "  • Color information is essential\n"
        "  • Power budget is not constrained"
    )
    ax_f.text(0.5, 0.5, rule_text, transform=ax_f.transAxes, fontsize=11,
              ha="center", va="center", family="monospace", linespacing=1.3,
              bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow",
                        edgecolor="#ccaa44", linewidth=2))
    ax_f.set_title("F. When to Use Each Sensor", fontweight="bold")

    fig.suptitle("Power-Efficient Tracking: DVS vs CIS Decision Analysis\n"
                 "Which sensor for which scene?",
                 fontsize=16, fontweight="bold", y=0.99)
    save(fig, "tracking_power_decision.png")


def plot_t7_temporal_tracking(merged_df):
    """Temporal variation: tracking error + power as object speeds up & slows down.
    Uses DVS and CIS temporal variation CSVs from Ish's and Harshita's models."""
    if not (os.path.isfile(DVS_TEMPORAL_CSV) and os.path.isfile(CIS_TEMPORAL_CSV)):
        print("  [skip] temporal variation CSVs not found")
        return

    dvs_tv = pd.read_csv(DVS_TEMPORAL_CSV)
    cis_tv = pd.read_csv(CIS_TEMPORAL_CSV)

    # use low_texture background for cleaner plot
    dvs_lt = dvs_tv[dvs_tv["background"] == "low_texture"].sort_values("time_s")
    cis_lt = cis_tv[cis_tv["background"] == "low_texture"].sort_values("time_s")

    # compute DVS tracking error at each time step (obj=50px assumed)
    obj_size = 50
    dvs_pos_error = dvs_lt["velocity_px_s"] / dvs_lt["event_rate_eff"]

    # compute CIS tracking error: use max_frame_rate from merged data
    max_fps_lt = merged_df[merged_df["background"] == "low_texture"]["max_frame_rate_hz"].iloc[0]
    cis_required_fps = dvs_lt["velocity_px_s"].values / (obj_size * 0.1)
    cis_actual_fps = np.minimum(cis_required_fps, max_fps_lt)
    cis_pos_error = 1.5 * dvs_lt["velocity_px_s"].values / cis_actual_fps
    cis_clipped = cis_required_fps > max_fps_lt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    time = dvs_lt["time_s"].values
    vel = dvs_lt["velocity_px_s"].values

    # ── top: velocity profile ──
    ax1.plot(time, vel, "k-", linewidth=2.5, marker="o", markersize=6)
    ax1.fill_between(time, 0, vel, alpha=0.08, color="gray")
    ax1.set_ylabel("Object Velocity (px/s)")
    ax1.set_title("Temporal Tracking: Object speeds up, peaks at 2000 px/s, slows down",
                   fontweight="bold", fontsize=13)
    ax1.set_ylim(0, 2200)

    # ── middle: tracking position error ──
    ax2.plot(time, dvs_pos_error, "o-", color=DVS_COLOR, linewidth=2.5,
             markersize=7, label="DVS position error", zorder=5)
    ax2.plot(time, cis_pos_error, "s-", color=CIS_COLOR, linewidth=2.5,
             markersize=7, label="CIS position error", zorder=4)

    # shade CIS-clipped time windows and add one labeled span for legend
    first_shade = True
    for i in range(len(time)):
        if cis_clipped[i]:
            lbl = "CIS fps clipped (can't keep up)" if first_shade else ""
            ax2.axvspan(time[i] - 0.4, time[i] + 0.4,
                         alpha=0.12, color="#d62728", zorder=0, label=lbl)
            first_shade = False

    ax2.set_ylabel("Position Error (px)")
    ax2.set_yscale("log")
    ax2.legend(fontsize=9, loc="center left")
    ratio = cis_pos_error.max() / dvs_pos_error.max()
    # place between the two lines (DVS ~0.003, CIS ~5) using axes coords
    ax2.text(0.97, 0.45,
             f"DVS: always < {dvs_pos_error.max():.3f} px\n"
             f"CIS: peaks at {cis_pos_error.max():.1f} px\n"
             f"({ratio:.0f}x worse at peak)",
             transform=ax2.transAxes, fontsize=9,
             verticalalignment="center", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                       edgecolor="#ccaa44", alpha=0.92, linewidth=1.2))

    # ── bottom: power over time ──
    ax3.plot(time, dvs_lt["power_total_mW"].values, "o-", color=DVS_COLOR,
             linewidth=2.5, markersize=7, label="DVS power")
    ax3.plot(time, cis_lt["power_mW"].values, "s-", color=CIS_COLOR,
             linewidth=2.5, markersize=7, label="CIS power")
    ax3.fill_between(time, dvs_lt["power_total_mW"].values, cis_lt["power_mW"].values,
                      alpha=0.08, color="gray")
    ax3.set_ylabel("Power (mW)")
    ax3.set_xlabel("Time (seconds)")
    ax3.legend(fontsize=10, loc="center left")
    add_finding(ax3,
                "DVS power scales with activity\n"
                "CIS power stays flat at 333 mW\n"
                "(reads all pixels every frame regardless)",
                loc="center right", fontsize=9)

    plt.tight_layout()
    save(fig, "tracking_temporal_variation.png")


def plot_t8_active_fraction(df):
    """Show what fraction of pixels are active at each velocity (DVS advantage)."""
    sub = df[df["threshold_name"] == "med_threshold"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for bg in BG_ORDER:
        bg_df = sub[(sub["background"] == bg) &
                     (sub["object_size_px"] == 50)].sort_values("velocity_px_s")
        ls = "-" if bg == "low_texture" else "--"
        ax.plot(bg_df["velocity_px_s"], bg_df["active_fraction"] * 100,
                marker="o", linestyle=ls, linewidth=2, markersize=7,
                label=f"DVS active — {BG_LABELS[bg]}")

    # CIS always reads 100% of pixels
    ax.axhline(y=100, color=CIS_COLOR, linewidth=2.5, linestyle="-",
               label="CIS (always reads 100% of pixels)")

    ax.set_xlabel("Velocity (px/s)")
    ax.set_ylabel("Active Pixel Fraction (%)")
    ax.set_title("Sensor Efficiency: DVS Active Pixels vs CIS Full-Frame Readout\n"
                 "(obj=50px, medium threshold)", fontweight="bold")
    ax.legend(fontsize=9, loc="center left")
    ax.set_ylim(-2, 110)

    add_finding(ax,
                "DVS: only 0.04–8% of pixels fire\n"
                "(rest stay silent, saving power)\n"
                "CIS: must read all 307,200 pixels\n"
                "every frame regardless of activity",
                loc="center right")
    fig.tight_layout()
    save(fig, "tracking_active_fraction.png")


def plot_t10_four_scenarios(merged_df):
    """4-scenario analysis: low/high clutter × static/temporal variation.
    Shows how background clutter and motion dynamics affect the DVS vs CIS tradeoff."""
    if not (os.path.isfile(DVS_TEMPORAL_CSV) and os.path.isfile(CIS_TEMPORAL_CSV)):
        print("  [skip] temporal variation CSVs not found")
        return

    dvs_tv = pd.read_csv(DVS_TEMPORAL_CSV)
    cis_tv = pd.read_csv(CIS_TEMPORAL_CSV)
    med50 = merged_df[(merged_df["threshold_name"] == "med_threshold") &
                       (merged_df["object_size_px"] == 50)]

    obj_size = 50
    max_fps = med50["max_frame_rate_hz"].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    scenarios = [
        ("low_texture",  "static",   "Low Clutter + Static Velocity"),
        ("high_texture", "static",   "High Clutter + Static Velocity"),
        ("low_texture",  "temporal", "Low Clutter + Temporal Variation"),
        ("high_texture", "temporal", "High Clutter + Temporal Variation"),
    ]

    for idx, (bg, mode, title) in enumerate(scenarios):
        ax = axes[idx // 2, idx % 2]

        if mode == "static":
            static = med50[med50["background"] == bg].sort_values("velocity_px_s")
            x_vals = static["velocity_px_s"].values
            x_label = "Velocity (px/s)"
            dvs_power = static["dvs_power_mW"].values
            cis_power = static["cis_power_mW"].values
            dvs_events = static["event_rate_eff"].values
            # compute tracking error
            dvs_err = x_vals / dvs_events
            cis_req_fps = x_vals / (obj_size * 0.1)
            cis_act_fps = np.minimum(cis_req_fps, max_fps)
            cis_err = 1.5 * x_vals / cis_act_fps
        else:
            dvs_t = dvs_tv[dvs_tv["background"] == bg].sort_values("time_s")
            cis_t = cis_tv[cis_tv["background"] == bg].sort_values("time_s")
            x_vals = dvs_t["time_s"].values
            x_label = "Time (s)"
            dvs_power = dvs_t["power_total_mW"].values
            cis_power = cis_t["power_mW"].values
            dvs_events = dvs_t["event_rate_eff"].values
            vel = dvs_t["velocity_px_s"].values
            dvs_err = vel / dvs_events
            cis_req_fps = vel / (obj_size * 0.1)
            cis_act_fps = np.minimum(cis_req_fps, max_fps)
            cis_err = 1.5 * vel / cis_act_fps

        # twin axes: power on left, tracking error on right
        color_pow = "#444444"
        ax.set_xlabel(x_label)

        # power bars/area
        ax.fill_between(x_vals, dvs_power, cis_power, alpha=0.08, color="gray")
        ax.plot(x_vals, cis_power, "s-", color=CIS_COLOR, linewidth=2.5,
                markersize=6, label="CIS power", zorder=4)
        ax.plot(x_vals, dvs_power, "o-", color=DVS_COLOR, linewidth=2.5,
                markersize=6, label="DVS power", zorder=5)
        ax.set_ylabel("Power (mW)")
        ax.set_ylim(0, max(cis_power) * 1.15)

        # annotate the power ratio at peak
        peak_idx = np.argmax(dvs_power)
        ratio_at_peak = cis_power[peak_idx] / dvs_power[peak_idx]
        ratio_min = min(cis_power / dvs_power)
        ax.annotate(f"{ratio_at_peak:.0f}x gap",
                     xy=(x_vals[peak_idx], (dvs_power[peak_idx] + cis_power[peak_idx]) / 2),
                     fontsize=10, fontweight="bold", color="gray", ha="center")

        # tracking error on right axis
        ax2 = ax.twinx()
        ax2.plot(x_vals, dvs_err, "^--", color="#2ca02c", linewidth=1.5,
                 markersize=5, alpha=0.8, label="DVS tracking error")
        ax2.plot(x_vals, cis_err, "v--", color="#d62728", linewidth=1.5,
                 markersize=5, alpha=0.8, label="CIS tracking error")
        ax2.set_ylabel("Tracking Error (px)")
        ax2.set_yscale("log")

        # combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=8, loc="center left", framealpha=0.9)

        # scenario summary box
        summary = (f"DVS: {dvs_power.min():.0f}–{dvs_power.max():.0f} mW\n"
                   f"CIS: {cis_power.min():.0f}–{cis_power.max():.0f} mW\n"
                   f"Ratio: {ratio_min:.0f}–{ratio_at_peak:.0f}x\n"
                   f"DVS err: {dvs_err.max():.4f} px\n"
                   f"CIS err: {cis_err.max():.1f} px")
        ax.text(0.97, 0.97, summary, transform=ax.transAxes, fontsize=8,
                va="top", ha="right", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="#ccaa44", alpha=0.92))

        ax.set_title(title, fontweight="bold", fontsize=12)

    fig.suptitle("4-Scenario Analysis: How Clutter and Motion Dynamics\n"
                 "Affect DVS vs CIS Power and Tracking",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "tracking_4_scenarios.png")


def plot_t11_background_activity(merged_df):
    """What happens when the background isn't static? Models independent
    background motion (wind, people, flicker) as additional DVS events."""
    med50_lt = merged_df[(merged_df["threshold_name"] == "med_threshold") &
                          (merged_df["object_size_px"] == 50) &
                          (merged_df["background"] == "low_texture")]

    n_pixels = 307200
    bg_pcts = np.array([0, 0.5, 1, 2, 5, 10, 20, 50, 100])
    velocities = [50, 200, 500, 2000]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # ── Left: DVS power vs background activity for different velocities ──
    for vel in velocities:
        row = med50_lt[med50_lt["velocity_px_s"] == vel].iloc[0]
        obj_events = row["event_rate_eff"]
        cis_pwr = row["cis_power_mW"]

        dvs_powers = []
        for bg_pct in bg_pcts:
            bg_events = (bg_pct / 100) * n_pixels * 10
            total = min(obj_events + bg_events, REFRACTORY_CAP)
            dvs_pwr = 19.14 + total * 4.95e-9 * 1e3
            dvs_powers.append(dvs_pwr)

        ax1.plot(bg_pcts, dvs_powers, "o-", linewidth=2, markersize=5,
                 label=f"DVS @ {vel} px/s")

    # CIS reference line (doesn't change with background activity)
    ax1.axhline(y=332.6, color=CIS_COLOR, linewidth=3, linestyle="-",
                label="CIS (constant, any background)")
    ax1.axhline(y=112, color="#d62728", linewidth=1.5, linestyle=":",
                alpha=0.6, label="DVS saturation cap (112 mW)")

    ax1.set_xlabel("Background Activity (%)\n"
                   "(0% = static room, 5% = busy indoor, 50% = extreme motion)",
                   fontsize=10)
    ax1.set_ylabel("Power (mW)")
    ax1.set_title("Power: DVS Stays Cheaper Even With Active Backgrounds",
                   fontweight="bold", fontsize=12)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_ylim(0, 400)
    ax1.set_xlim(-2, 105)

    # ── Right: Signal-to-noise ratio (tracking quality) vs bg activity ──
    for vel in velocities:
        row = med50_lt[med50_lt["velocity_px_s"] == vel].iloc[0]
        obj_events = row["event_rate_eff"]

        snr_vals = []
        for bg_pct in bg_pcts:
            bg_events = (bg_pct / 100) * n_pixels * 10
            # signal = object events, noise = background events + false positives
            signal = obj_events
            noise = bg_events + obj_events * FALSE_POS_RATE
            snr = signal / (signal + noise) * 100  # signal fraction %
            snr_vals.append(snr)

        ax2.plot(bg_pcts, snr_vals, "o-", linewidth=2, markersize=5,
                 label=f"DVS @ {vel} px/s")

    # CIS doesn't have this problem -- full frame always available
    ax2.axhline(y=100, color=CIS_COLOR, linewidth=3, linestyle="-",
                label="CIS (full frame, no event noise)")
    ax2.axhline(y=50, color="gray", linewidth=1, linestyle=":",
                alpha=0.6)
    ax2.text(102, 50, "50%\n(noise\ndominates)", fontsize=8, color="gray", va="center")

    ax2.set_xlabel("Background Activity (%)\n"
                   "(0% = static room, 5% = busy indoor, 50% = extreme motion)",
                   fontsize=10)
    ax2.set_ylabel("DVS Signal Fraction (%)\n(object events / total events)")
    ax2.set_title("Tracking Quality: Where Background Noise Hurts DVS",
                   fontweight="bold", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_ylim(0, 105)
    ax2.set_xlim(-2, 105)

    add_finding(ax2,
                "DVS tradeoff: active backgrounds\n"
                "flood events with noise, degrading\n"
                "tracking signal-to-noise ratio.\n"
                "CIS captures full frame → can\n"
                "filter spatially (bg subtraction).",
                loc="lower left")

    fig.suptitle("The Real DVS vs CIS Tradeoff: Background Activity\n"
                 "DVS always wins on power -- but loses on signal quality in noisy scenes",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    save(fig, "tracking_background_activity.png")


# ── main ───────────────────────────────────────────────────────────────

def main():
    print("Loading merged data...")
    df = pd.read_csv(MERGED_CSV)
    print(f"  {len(df)} rows loaded from merged_cis_dvs_data.csv")

    print("\nComputing tracking metrics...")
    result = compute_all_tracking(df)

    print("\nSaving results...")
    save_results(result)

    print("\nGenerating plots...")
    plot_t1_position_error_vs_velocity(result)
    plot_t2_missed_fraction(result)
    plot_t3_tracking_quality(result)
    plot_t4_tracking_heatmap(result)
    plot_t5_error_vs_sensor_params(result)
    plot_t6_slide_summary(result)
    plot_t7_temporal_tracking(result)
    plot_t8_active_fraction(result)
    plot_t9_power_tracking_decision(result)
    plot_t10_four_scenarios(result)
    plot_t11_background_activity(result)

    # print summary
    med = result[result["threshold_name"] == "med_threshold"]
    print("\n-- Tracking Baseline Summary --")
    print(f"  DVS position error range: "
          f"{med['dvs_position_error_px'].min():.4f} – "
          f"{med['dvs_position_error_px'].max():.4f} px")
    print(f"  CIS position error range: "
          f"{med['cis_position_error_px'].min():.4f} – "
          f"{med['cis_position_error_px'].max():.4f} px")
    print(f"  DVS tracking quality:     "
          f"{med['dvs_tracking_quality'].min():.4f} – "
          f"{med['dvs_tracking_quality'].max():.4f}")
    print(f"  CIS tracking quality:     "
          f"{med['cis_tracking_quality'].min():.4f} – "
          f"{med['cis_tracking_quality'].max():.4f}")
    dvs_sat = (result["dvs_missed_fraction"] > 0).sum()
    cis_miss = (result["cis_missed_fraction"] > 0).sum()
    print(f"  DVS saturated rows:       {dvs_sat}/{len(result)}")
    print(f"  CIS fps-clipped rows:     {cis_miss}/{len(result)}")

    # new model features
    dvs_track_all = (result["dvs_can_track"] == 1).all()
    print(f"\n  dvs_can_track flag:       {'ALL scenes trackable' if dvs_track_all else 'SOME scenes NOT trackable'}")
    print(f"  Active pixel fraction:    "
          f"{med['active_fraction'].min()*100:.2f}% – "
          f"{med['active_fraction'].max()*100:.2f}%")
    print(f"  (CIS reads 100% of pixels every frame)")
    print("\nTracking baseline complete.")


if __name__ == "__main__":
    main()
