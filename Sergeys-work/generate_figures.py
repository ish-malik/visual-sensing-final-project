"""Reads the sweep CSVs and writes F1..F7 into results/figures/."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

sys.path.insert(0, HERE)
from models.sensor_database import DVS_SENSORS, CIS_SENSORS

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    }
)

CIS_COLORS = {
    "OV7251 (OmniVision)": "#1f77b4",
    "IMX327 (Sony)": "#2ca02c",
    "AR0234 (ON Semi)": "#17becf",
    "IMX462 (Sony)": "#9467bd",
}
DVS_COLORS = {
    "Lichtsteiner 2008": "#d62728",
    "DAVIS346": "#ff7f0e",
    "Samsung DVS-Gen3.1": "#e377c2",
    "Prophesee IMX636": "#8c564b",
}

# Drop vendor, tag modality. CSV sensor_name stays authoritative.
DISPLAY_NAMES = {
    "OV7251 (OmniVision)": "OV7251 (CIS)",
    "IMX327 (Sony)": "IMX327 (CIS)",
    "AR0234 (ON Semi)": "AR0234 (CIS)",
    "IMX462 (Sony)": "IMX462 (CIS)",
    "Lichtsteiner 2008": "Lichtsteiner (DVS)",
    "DAVIS346": "DAVIS346 (DVS)",
    "Samsung DVS-Gen3.1": "DVS-Gen3.1 (DVS)",
    "Prophesee IMX636": "IMX636 (DVS)",
}


def _dn(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _load_sweep_a() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, "sweep_a_analytical.csv"))


def _load_sweep_b() -> pd.DataFrame:
    p = os.path.join(RESULTS, "sweep_b_mot17.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def _save(fig, name: str):
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


def fig01_sensor_table():
    """DVS + CIS rundown table."""
    rows = []
    for s in DVS_SENSORS:
        res = f"{s.resolution[0]}x{s.resolution[1]}"
        rows.append(
            [
                _dn(s.name),
                "DVS",
                res,
                f"{s.p_static_mw:.0f} mW static",
                f"{s.e_per_event_nj:.2f} nJ/event",
                f"{s.pixel_latency_us:.0f} us latency",
                f"{s.max_event_rate_mevps:.0f} Mev/s cap",
            ]
        )
    for s in CIS_SENSORS:
        res = f"{s.resolution[0]}x{s.resolution[1]}"
        rows.append(
            [
                _dn(s.name),
                "CIS",
                res,
                f"{s.power_idle_mw:.0f} mW idle",
                f"{s.power_at_max_fps_mw:.0f} mW @ max fps",
                f"{s.max_fps:.0f} fps max",
                f"{s.adc_bits} bit ADC",
            ]
        )

    col_labels = [
        "Sensor",
        "Type",
        "Resolution",
        "Static / idle",
        "Dynamic / peak",
        "Speed spec",
        "Extra",
    ]

    fig, ax = plt.subplots(figsize=(14, 0.55 * (len(rows) + 2)))
    ax.axis("off")
    fig.suptitle("Sensor rundown: DVS and CIS used across figures", fontweight="bold")

    tbl = ax.table(
        cellText=rows, colLabels=col_labels, loc="center", cellLoc="left", colLoc="left"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.4)

    for i, row in enumerate(rows, start=1):
        bg = "#fde2e1" if row[1] == "DVS" else "#dce7f5"
        for col in range(len(col_labels)):
            tbl[(i, col)].set_facecolor(bg)
    for col in range(len(col_labels)):
        tbl[(0, col)].set_facecolor("#222222")
        tbl[(0, col)].set_text_props(color="white", fontweight="bold")

    plt.tight_layout()
    _save(fig, "fig01_sensor_table.png")


def fig02_crossover(df_a: pd.DataFrame, obj_size_px: int = 50):
    """Power vs velocity, stars mark v*."""
    df = df_a[df_a["obj_size_px"] == obj_size_px].copy()
    all_bg = sorted(df["bg_density"].unique())
    bg_vals = [all_bg[0], all_bg[-1]]  # only the extremes
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle(
        f"CIS vs DVS Power and Velocity Crossover (obj {obj_size_px}px) — "
        f"v* marks where DVS becomes lower power than CIS",
        fontweight="bold",
    )

    for ax, bg in zip(axes, bg_vals):
        bg_label = df[df["bg_density"] == bg]["bg_label"].iloc[0]
        ax.set_title(f"{bg_label} (d {bg})", fontweight="bold")

        for cis in CIS_SENSORS:
            c = CIS_COLORS[cis.name]
            for policy, ls, alpha in [("adaptive", "-", 1.0), ("locked", "--", 0.55)]:
                sub = df[
                    (df["sensor_name"] == cis.name)
                    & (df["bg_density"] == bg)
                    & (df["operating_point"] == policy)
                ].sort_values("velocity_px_s")
                if len(sub):
                    ax.plot(
                        sub["velocity_px_s"],
                        sub["power_mw"],
                        color=c,
                        linestyle=ls,
                        linewidth=2.2,
                        alpha=alpha,
                        marker="o" if ls == "-" else "s",
                        markersize=5,
                    )

        for dvs in DVS_SENSORS:
            c = DVS_COLORS[dvs.name]
            for theta, ls, lw, alpha in [
                (0.20, "-", 2.2, 1.0),
                (0.05, ":", 1.0, 0.5),
                (0.10, ":", 1.0, 0.5),
                (0.40, ":", 1.0, 0.5),
            ]:
                sub = df[
                    (df["sensor_name"] == dvs.name)
                    & (df["bg_density"] == bg)
                    & (df["operating_point"] == f"theta={theta:.2f}")
                ].sort_values("velocity_px_s")
                if len(sub):
                    ax.plot(
                        sub["velocity_px_s"],
                        sub["power_mw"],
                        color=c,
                        linestyle=ls,
                        linewidth=lw,
                        alpha=alpha,
                        marker="^" if ls == "-" else None,
                        markersize=5,
                    )

        v_star_values = []
        for cis in CIS_SENSORS:
            sub_cis = df[
                (df["sensor_name"] == cis.name)
                & (df["bg_density"] == bg)
                & (df["operating_point"] == "adaptive")
            ].sort_values("velocity_px_s")
            for dvs in DVS_SENSORS:
                sub_dvs = df[
                    (df["sensor_name"] == dvs.name)
                    & (df["bg_density"] == bg)
                    & (df["operating_point"] == "theta=0.20")
                ].sort_values("velocity_px_s")
                if len(sub_cis) == 0 or len(sub_dvs) == 0:
                    continue
                v = sub_cis["velocity_px_s"].to_numpy()
                p_cis = sub_cis["power_mw"].to_numpy()
                p_dvs = sub_dvs["power_mw"].to_numpy()
                diff = p_cis - p_dvs
                sign = np.sign(diff)
                crossings = np.where(np.diff(sign) != 0)[0]
                for idx in crossings:
                    v0, v1 = v[idx], v[idx + 1]
                    d0, d1 = diff[idx], diff[idx + 1]
                    if d1 == d0:
                        continue
                    v_star = v0 - d0 * (v1 - v0) / (d1 - d0)
                    # interp power so the star lands on the curve
                    p_star = p_cis[idx] + (p_cis[idx + 1] - p_cis[idx]) * (
                        (v_star - v0) / (v1 - v0)
                    )
                    v_star_values.append(v_star)
                    ax.plot(
                        v_star,
                        p_star,
                        marker="*",
                        markersize=16,
                        color="black",
                        markerfacecolor="gold",
                        markeredgewidth=1.2,
                        zorder=6,
                    )

        if v_star_values:
            v_lo = min(v_star_values)
            v_hi = max(v_star_values)
            v_med = float(np.median(v_star_values))
            ax.axvspan(v_lo, v_hi, color="#ffd27f", alpha=0.25, zorder=0)
            ax.axvline(
                v_med,
                color="#b36b00",
                linestyle="-",
                linewidth=2.0,
                alpha=0.85,
                zorder=3,
            )
            ymin, ymax = ax.get_ylim() if ax.get_ylim()[1] > 0 else (1, 1e4)
            ax.annotate(
                f"v* ~ {v_med:.0f} px/s\nDVS wins →",
                xy=(v_med, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(6, -18),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="#b36b00",
                bbox=dict(
                    facecolor="white",
                    edgecolor="#b36b00",
                    boxstyle="round,pad=0.25",
                    alpha=0.9,
                ),
            )
            ax.annotate(
                "← CIS wins",
                xy=(v_med, 0.02),
                xycoords=("data", "axes fraction"),
                xytext=(-70, 0),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="#1f4f8a",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Velocity (px/s)")
        ax.set_ylabel("Total power (mW)")
        ax.grid(True, which="both", alpha=0.3)

    handles = []
    for c in CIS_SENSORS:
        handles.append(
            Line2D(
                [],
                [],
                color=CIS_COLORS[c.name],
                linestyle="-",
                marker="o",
                label=_dn(c.name),
            )
        )
    for d in DVS_SENSORS:
        handles.append(
            Line2D(
                [],
                [],
                color=DVS_COLORS[d.name],
                linestyle="-",
                marker="^",
                label=_dn(d.name),
            )
        )
    handles.append(Line2D([], [], color="gray", linestyle="-", label="CIS adaptive"))
    handles.append(Line2D([], [], color="gray", linestyle="--", label="CIS locked"))
    handles.append(
        Line2D([], [], color="gray", linestyle=":", label="DVS theta 0.05, 0.10, 0.40")
    )
    handles.append(
        Line2D(
            [],
            [],
            color="black",
            marker="*",
            markersize=14,
            markerfacecolor="gold",
            linestyle="None",
            label="Crossover v* (CIS = DVS power)",
        )
    )
    handles.append(
        Line2D(
            [],
            [],
            color="#b36b00",
            linestyle="-",
            linewidth=2.0,
            label="Median v* across CIS/DVS pairs",
        )
    )

    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    _save(fig, "fig02_crossover_power_vs_velocity.png")


def _compute_vstar_table(df_a: pd.DataFrame, obj_size_px: int) -> pd.DataFrame:
    rows = []
    bg_values = sorted(df_a["bg_density"].unique())
    for cis in CIS_SENSORS:
        for dvs in DVS_SENSORS:
            for bg in bg_values:
                sub_cis = df_a[
                    (df_a["sensor_name"] == cis.name)
                    & (df_a["obj_size_px"] == obj_size_px)
                    & (df_a["bg_density"] == bg)
                    & (df_a["operating_point"] == "adaptive")
                ].sort_values("velocity_px_s")
                sub_dvs = df_a[
                    (df_a["sensor_name"] == dvs.name)
                    & (df_a["obj_size_px"] == obj_size_px)
                    & (df_a["bg_density"] == bg)
                    & (df_a["operating_point"] == "theta=0.20")
                ].sort_values("velocity_px_s")
                if len(sub_cis) == 0 or len(sub_dvs) == 0:
                    continue
                v = sub_cis["velocity_px_s"].to_numpy()
                diff = sub_cis["power_mw"].to_numpy() - sub_dvs["power_mw"].to_numpy()
                sign = np.sign(diff)
                idxs = np.where(np.diff(sign) != 0)[0]
                if len(idxs) == 0:
                    v_star = np.nan
                    dominant = "DVS" if diff[0] > 0 else "CIS"
                else:
                    idx = idxs[0]
                    v0, v1 = v[idx], v[idx + 1]
                    d0, d1 = diff[idx], diff[idx + 1]
                    v_star = v0 - d0 * (v1 - v0) / (d1 - d0) if d1 != d0 else v0
                    dominant = "mixed"
                rows.append(
                    {
                        "cis_sensor": cis.name,
                        "dvs_sensor": dvs.name,
                        "bg_density": bg,
                        "v_star": v_star,
                        "dominant": dominant,
                    }
                )
    return pd.DataFrame(rows)


def fig03_vstar_vs_background(df_a: pd.DataFrame, obj_size_px: int = 50):
    """v* vs background density, one panel per CIS sensor."""
    v_df = _compute_vstar_table(df_a, obj_size_px)
    v_df.to_csv(os.path.join(RESULTS, "vstar_table.csv"), index=False)

    bg_values = sorted(v_df["bg_density"].unique())
    cis_names = sorted(v_df["cis_sensor"].unique())
    dvs_names = sorted(v_df["dvs_sensor"].unique())

    fig, axes = plt.subplots(
        1, len(cis_names), figsize=(4.8 * len(cis_names), 6.2), sharey=True
    )
    if len(cis_names) == 1:
        axes = [axes]
    fig.suptitle(
        f"Crossover velocity v* vs background density (obj {obj_size_px}px) — "
        f"DVS wins for any velocity above its line",
        fontweight="bold",
    )

    finite_all = v_df["v_star"].dropna()
    y_max = float(finite_all.max()) if len(finite_all) else 10.0
    y_top = y_max * 1.15 if y_max > 0 else 10.0

    for ax, cis_name in zip(axes, cis_names):
        for dvs_name in dvs_names:
            sub = v_df[
                (v_df["cis_sensor"] == cis_name) & (v_df["dvs_sensor"] == dvs_name)
            ].sort_values("bg_density")
            if len(sub) == 0:
                continue
            finite_mask = sub["v_star"].notna()
            finite_sub = sub[finite_mask]
            color = DVS_COLORS[dvs_name]
            if len(finite_sub):
                ax.plot(
                    finite_sub["bg_density"],
                    finite_sub["v_star"],
                    marker="o",
                    color=color,
                    linewidth=2.2,
                    markersize=7,
                    label=_dn(dvs_name),
                    zorder=4,
                )
                ax.fill_between(
                    finite_sub["bg_density"],
                    finite_sub["v_star"],
                    y_top,
                    color=color,
                    alpha=0.06,
                    zorder=1,
                )
            # bg points where DVS dominates even at v=0
            miss_sub = sub[~finite_mask]
            for _, r in miss_sub.iterrows():
                ax.scatter(r["bg_density"], 0, marker="x", color=color, s=60)

        ax.set_title(_dn(cis_name), fontsize=10, fontweight="bold")
        ax.set_xlabel("Background density d")
        ax.set_ylabel("v* (px/s)  —  DVS wins above, CIS wins below")
        ax.set_xticks(bg_values)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        ax.text(
            0.02,
            0.96,
            "DVS wins ↑",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color="#444",
            va="top",
        )
        ax.text(
            0.02,
            0.04,
            "CIS wins ↓",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color="#444",
            va="bottom",
        )

    for ax in axes:
        ax.set_ylim(bottom=-0.05 * max(y_top, 10), top=y_top)

    plt.tight_layout()
    _save(fig, "fig03_vstar_vs_background.png")


def fig04_mot17_validation(df_b: pd.DataFrame):
    """HOTA / DetA / AssA / MOTA vs power on MOT17."""
    if len(df_b) == 0:
        print("[F4] No Sweep B data, skipping")
        return
    df_b = df_b[df_b["source"] == "mot17"]
    agg = (
        df_b.groupby(["sensor_name", "sensor_type", "operating_point", "coast"])
        .agg(
            power_mw=("power_mw", "mean"),
            hota_mean=("hota", "mean"),
            hota_std=("hota", "std"),
            det_a_mean=("det_a", "mean"),
            det_a_std=("det_a", "std"),
            ass_a_mean=("ass_a", "mean"),
            ass_a_std=("ass_a", "std"),
            mota_mean=("mota", "mean"),
            mota_std=("mota", "std"),
        )
        .reset_index()
    )
    for col in ["hota_std", "det_a_std", "ass_a_std", "mota_std"]:
        agg[col] = agg[col].fillna(0.0)

    # Apples-to-apples: DVS without coasting.
    dvs_real = agg[(agg["sensor_type"] == "DVS") & (~agg["coast"])]
    cis = agg[agg["sensor_type"] == "CIS"]

    all_power = np.concatenate(
        [dvs_real["power_mw"].to_numpy(), cis["power_mw"].to_numpy()]
    )
    if len(all_power) == 0:
        print("[F4] No power rows, skipping")
        return
    x_lo = max(1.0, float(all_power.min()) * 0.6)
    x_hi = float(all_power.max()) * 1.6

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
    fig.suptitle(
        "MOT17 validation (real pedestrians): HOTA / DetA / AssA / MOTA vs power",
        fontweight="bold",
        y=0.995,
    )

    def _plot(ax, subset, metric_mean, metric_std, marker, color_map):
        # MOTA can go negative on crowded MOT17 — clip at 0, annotate the real value.
        for sensor_name, grp in subset.groupby("sensor_name"):
            c = color_map.get(sensor_name, "gray")
            grp_sorted = grp.sort_values("power_mw")
            y_true = grp_sorted[metric_mean].to_numpy()
            y_disp = np.clip(y_true, 0.0, None)
            ax.errorbar(
                grp_sorted["power_mw"],
                y_disp,
                yerr=grp_sorted[metric_std],
                marker=marker,
                linestyle="-",
                color=c,
                linewidth=1.6,
                capsize=3,
                markersize=9,
                label=_dn(sensor_name),
            )
            for (_, r), y_t in zip(grp_sorted.iterrows(), y_true):
                ax.annotate(
                    r["operating_point"],
                    (r["power_mw"], max(y_t, 0.0)),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=7,
                )
                if y_t < 0:
                    ax.annotate(
                        f"{y_t:.2f}",
                        (r["power_mw"], 0.0),
                        textcoords="offset points",
                        xytext=(0, -10),
                        ha="center",
                        fontsize=7,
                        color="#a00",
                        fontweight="bold",
                    )

    panels = [
        (axes[0, 0], "HOTA overall score", "hota_mean", "hota_std"),
        (axes[0, 1], "DetA — did you find them", "det_a_mean", "det_a_std"),
        (axes[1, 0], "AssA — did IDs stay", "ass_a_mean", "ass_a_std"),
        (axes[1, 1], "MOTA legacy metric", "mota_mean", "mota_std"),
    ]

    for ax, title, mean_col, std_col in panels:
        _plot(ax, dvs_real, mean_col, std_col, "^", DVS_COLORS)
        _plot(ax, cis, mean_col, std_col, "o", CIS_COLORS)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Power (mW)")
        ax.set_ylabel(title.split(" — ")[0].split(" legacy")[0].split(" overall")[0])
        ax.set_xscale("log")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.0, 1.0)
        if mean_col == "mota_mean":
            ax.axhline(y=0, color="black", linewidth=0.8, linestyle=":")
        ax.grid(True, alpha=0.3)

    handles = []
    for c in CIS_SENSORS:
        handles.append(
            Line2D(
                [],
                [],
                color=CIS_COLORS[c.name],
                marker="o",
                linestyle="-",
                label=_dn(c.name),
            )
        )
    for d in DVS_SENSORS:
        handles.append(
            Line2D(
                [],
                [],
                color=DVS_COLORS[d.name],
                marker="^",
                linestyle="-",
                label=_dn(d.name),
            )
        )
    handles.append(
        Line2D(
            [],
            [],
            color="gray",
            marker="o",
            linestyle="None",
            label="CIS sensors (circles)",
        )
    )
    handles.append(
        Line2D(
            [],
            [],
            color="gray",
            marker="^",
            linestyle="None",
            label="DVS sensors (triangles)",
        )
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    _save(fig, "fig04_mot17_validation.png")


def fig05_coasting_comparison(df_b: pd.DataFrame):
    """DVS HOTA/MOTA: coast on vs off, per threshold."""
    if len(df_b) == 0:
        print("[F5] No Sweep B data, skipping")
        return
    df_b = df_b[df_b["source"] == "mot17"]
    dvs = df_b[df_b["sensor_type"] == "DVS"].copy()
    if len(dvs) == 0:
        print("[F5] No DVS rows, skipping")
        return

    agg = (
        dvs.groupby(["sensor_name", "theta", "coast"])
        .agg(
            hota_mean=("hota", "mean"),
            hota_std=("hota", "std"),
            mota_mean=("mota", "mean"),
            mota_std=("mota", "std"),
        )
        .reset_index()
    )
    agg["hota_std"] = agg["hota_std"].fillna(0.0)
    agg["mota_std"] = agg["mota_std"].fillna(0.0)

    sensors = sorted(agg["sensor_name"].unique())
    fig, axes = plt.subplots(
        2, len(sensors), figsize=(5 * len(sensors), 10), sharey=True
    )
    if len(sensors) == 1:
        axes = np.array(axes).reshape(2, 1)
    fig.suptitle(
        "DVS coasting comparison: HOTA and MOTA with vs without coast",
        fontweight="bold",
    )

    thetas = sorted(agg["theta"].unique())
    x = np.arange(len(thetas))
    bar_w = 0.35

    for col, sensor_name in enumerate(sensors):
        sub = agg[agg["sensor_name"] == sensor_name]

        for row_idx, (metric_mean, metric_std, label) in enumerate(
            [
                ("hota_mean", "hota_std", "HOTA"),
                ("mota_mean", "mota_std", "MOTA"),
            ]
        ):
            ax = axes[row_idx, col]
            off_vals, off_stds, on_vals, on_stds = [], [], [], []
            for t in thetas:
                off = sub[(sub["theta"] == t) & (sub["coast"] == False)]
                on = sub[(sub["theta"] == t) & (sub["coast"] == True)]
                off_vals.append(off[metric_mean].iloc[0] if len(off) else 0)
                off_stds.append(off[metric_std].iloc[0] if len(off) else 0)
                on_vals.append(on[metric_mean].iloc[0] if len(on) else 0)
                on_stds.append(on[metric_std].iloc[0] if len(on) else 0)

            off_disp = [max(v, 0.0) for v in off_vals]
            on_disp = [max(v, 0.0) for v in on_vals]

            ax.bar(
                x - bar_w / 2,
                off_disp,
                bar_w,
                yerr=off_stds,
                color="#2255AA",
                label="coast off",
                edgecolor="black",
                linewidth=0.8,
                capsize=4,
            )
            ax.bar(
                x + bar_w / 2,
                on_disp,
                bar_w,
                yerr=on_stds,
                color="#cc4444",
                label="coast on",
                edgecolor="black",
                linewidth=0.8,
                capsize=4,
            )

            for xi, v in zip(x - bar_w / 2, off_vals):
                if v < 0:
                    ax.annotate(
                        f"{v:.2f}",
                        (xi, 0.0),
                        textcoords="offset points",
                        xytext=(0, 12),
                        ha="center",
                        fontsize=7,
                        color="#1a3c7a",
                        fontweight="bold",
                    )
            for xi, v in zip(x + bar_w / 2, on_vals):
                if v < 0:
                    ax.annotate(
                        f"{v:.2f}",
                        (xi, 0.0),
                        textcoords="offset points",
                        xytext=(0, 12),
                        ha="center",
                        fontsize=7,
                        color="#8a2e2e",
                        fontweight="bold",
                    )

            ax.set_xticks(x)
            ax.set_xticklabels([f"theta {t:.2f}" for t in thetas])
            if row_idx == 0:
                ax.set_title(_dn(sensor_name), fontsize=10, fontweight="bold")
            ax.set_ylabel(label)
            ax.set_ylim(0.0, 1.0)
            if row_idx == 1:
                ax.axhline(y=0, color="black", linewidth=0.8, linestyle=":")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig05_coasting_comparison.png")


def fig06_modulcis_calibration():
    """Linear CIS power vs ModuCIS SPICE, scatter."""
    from models.cis_spice_lut import CisSpiceLut

    lut_path = os.path.join(RESULTS, "cis_spice_lut.json")
    if not os.path.exists(lut_path):
        print("[F6] No cis_spice_lut.json, skipping")
        return
    lut = CisSpiceLut.load(lut_path)

    rows = []
    for cis in CIS_SENSORS:
        for (w, h, fps_i, bits), spice_mw in lut.entries.items():
            if (int(w), int(h)) != cis.resolution or int(bits) != cis.adc_bits:
                continue
            linear_mw = cis.power_mw(float(fps_i))
            rows.append(
                {
                    "sensor": cis.name,
                    "fps": int(fps_i),
                    "spice": float(spice_mw),
                    "linear": float(linear_mw),
                }
            )

    if not rows:
        print("[F6] No matching CIS entries in LUT, skipping")
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle("ModuCIS SPICE vs linear interp calibration", fontweight="bold")

    for sensor_name, grp in df.groupby("sensor"):
        ax.scatter(
            grp["linear"],
            grp["spice"],
            s=80,
            alpha=0.85,
            edgecolor="black",
            color=CIS_COLORS.get(sensor_name, "gray"),
            label=_dn(sensor_name),
        )
        for _, r in grp.iterrows():
            ax.annotate(
                f'{r["fps"]}fps',
                (r["linear"], r["spice"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    lo = min(df["linear"].min(), df["spice"].min())
    hi = max(df["linear"].max(), df["spice"].max())
    ax.plot(
        [lo, hi],
        [lo, hi],
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="y equals x",
    )

    ax.set_xlabel("Linear interp power (mW) from sensor_database")
    ax.set_ylabel("ModuCIS SPICE power (mW) from CIS_Array")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig06_modulcis_calibration.png")


def fig07_mot17_vs_synthetic(df_b: pd.DataFrame):
    """HOTA/MOTA on MOT17 vs synthetic. Gap = tracker headroom lost to crowding."""
    if len(df_b) == 0:
        print("[F7] No Sweep B data, skipping")
        return
    have_sources = sorted(df_b["source"].dropna().unique().tolist())
    if "mot17" not in have_sources or not any(
        s.startswith("synthetic") for s in have_sources
    ):
        print("[F7] Missing MOT17 or synthetic rows, skipping")
        return

    dfb = df_b[df_b["coast"] == False].copy()

    agg = (
        dfb.groupby(["sensor_name", "sensor_type", "operating_point", "source"])
        .agg(
            power_mw=("power_mw", "mean"),
            hota_mean=("hota", "mean"),
            hota_std=("hota", "std"),
            mota_mean=("mota", "mean"),
            mota_std=("mota", "std"),
        )
        .reset_index()
    )
    agg["hota_std"] = agg["hota_std"].fillna(0.0)
    agg["mota_std"] = agg["mota_std"].fillna(0.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "MOT17 real video vs synthetic scenes, HOTA and MOTA by source",
        fontweight="bold",
    )

    source_order = ["mot17"] + sorted(
        s for s in have_sources if s.startswith("synthetic")
    )
    source_colors = {
        "mot17": "#222222",
        "synthetic_low": "#2ca02c",
        "synthetic_high": "#d62728",
    }
    source_markers = {
        "mot17": "o",
        "synthetic_low": "s",
        "synthetic_high": "D",
    }

    sensor_panels = [
        (axes[0, 0], "DVS", "hota_mean", "hota_std", "HOTA"),
        (axes[0, 1], "CIS", "hota_mean", "hota_std", "HOTA"),
        (axes[1, 0], "DVS", "mota_mean", "mota_std", "MOTA"),
        (axes[1, 1], "CIS", "mota_mean", "mota_std", "MOTA"),
    ]

    y_lo, y_hi = 0.0, 1.0

    for ax, sensor_type, mean_col, std_col, metric_label in sensor_panels:
        sub = agg[agg["sensor_type"] == sensor_type]
        sensor_names = sorted(sub["sensor_name"].unique())
        x_bases = np.arange(len(sensor_names))
        bar_w = 0.22

        for i, source in enumerate(source_order):
            src_sub = sub[sub["source"] == source]
            vals = []
            errs = []
            for name in sensor_names:
                rows = src_sub[src_sub["sensor_name"] == name]
                vals.append(rows[mean_col].mean() if len(rows) else 0.0)
                errs.append(rows[std_col].mean() if len(rows) else 0.0)
            offsets = x_bases + (i - (len(source_order) - 1) / 2) * bar_w
            # clamp at 0 for display, annotate real value if MOTA is negative
            display_vals = [max(v, 0.0) for v in vals]
            ax.bar(
                offsets,
                display_vals,
                bar_w,
                yerr=errs,
                color=source_colors.get(source, "#888888"),
                edgecolor="black",
                linewidth=0.8,
                capsize=3,
                label=source,
            )
            for off, v in zip(offsets, vals):
                if v < 0:
                    ax.annotate(
                        f"{v:.2f}",
                        (off, 0.0),
                        textcoords="offset points",
                        xytext=(0, 14),
                        ha="center",
                        fontsize=7,
                        color="#a00",
                        fontweight="bold",
                    )

        ax.set_xticks(x_bases)
        ax.set_xticklabels(
            [_dn(n) for n in sensor_names], rotation=18, ha="right", fontsize=8
        )
        ax.set_ylabel(f"{metric_label} (mean)")
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"{sensor_type} — {metric_label}", fontweight="bold")
        if metric_label == "MOTA":
            ax.axhline(y=0, color="black", linewidth=0.8, linestyle=":")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    _save(fig, "fig07_mot17_vs_synthetic.png")


def main():
    df_a = _load_sweep_a()
    df_b = _load_sweep_b()

    print(f"[Figures] Sweep A rows: {len(df_a)}")
    print(f"[Figures] Sweep B rows: {len(df_b)}")

    fig01_sensor_table()
    fig02_crossover(df_a)
    fig03_vstar_vs_background(df_a)
    fig04_mot17_validation(df_b)
    fig05_coasting_comparison(df_b)
    fig06_modulcis_calibration()
    fig07_mot17_vs_synthetic(df_b)
    print("\n[Figures] done")


if __name__ == "__main__":
    main()
