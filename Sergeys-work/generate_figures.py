"""Reads the sweep CSVs and writes the slide figures to results/figures/.

Run after run_sweeps.py. F1 is the pipeline architecture Mermaid graph
in results/fig01_pipeline_architecture.md. F2 is the main power vs
velocity crossover plot. F3 shows V star as a continuous line versus
background density for each sensor pair. F4 is the MOT17 validation for
both HOTA and MOTA. F5 is the coasting on vs off comparison. F6 checks
the CIS linear power model against ModuCIS SPICE. F7 compares the MOT17
runs against the synthetic scene runs so we can see how much harder real
video is than the analytical model assumes. The design rule table lives
in results/design_rule.md.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE   = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
FIGS    = os.path.join(RESULTS, 'figures')
os.makedirs(FIGS, exist_ok=True)

sys.path.insert(0, HERE)
from sensor_database import DVS_SENSORS, CIS_SENSORS

plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'font.size':         10,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'legend.fontsize':   9,
    'figure.titlesize':  14,
})

CIS_COLORS = {
    'OV7251 (OmniVision)': '#1f77b4',
    'IMX327 (Sony)':       '#2ca02c',
    'AR0234 (ON Semi)':    '#17becf',
    'IMX462 (Sony)':       '#9467bd',
}
DVS_COLORS = {
    'Lichtsteiner 2008':    '#d62728',
    'DAVIS346':             '#ff7f0e',
    'Samsung DVS-Gen3.1':   '#e377c2',
    'Prophesee IMX636':     '#8c564b',
}


def _load_sweep_a() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, 'sweep_a_analytical.csv'))


def _load_sweep_b() -> pd.DataFrame:
    p = os.path.join(RESULTS, 'sweep_b_mot17.csv')
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def _save(fig, name: str):
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {path}')


def fig02_crossover(df_a: pd.DataFrame, obj_size_px: int = 50):
    """Main power vs velocity plot with CIS and DVS curves and crossover marks."""
    df = df_a[df_a['obj_size_px'] == obj_size_px].copy()
    all_bg = sorted(df['bg_density'].unique())
    bg_vals = [all_bg[0], all_bg[-1]]  # show only the extremes (low and high)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle(f'CIS vs DVS Power and Velocity Crossover (obj {obj_size_px}px)',
                 fontweight='bold')

    for ax, bg in zip(axes, bg_vals):
        bg_label = df[df['bg_density'] == bg]['bg_label'].iloc[0]
        ax.set_title(f'{bg_label} (d {bg})', fontweight='bold')

        for cis in CIS_SENSORS:
            c = CIS_COLORS[cis.name]
            for policy, ls, alpha in [('adaptive', '-', 1.0), ('locked', '--', 0.55)]:
                sub = df[(df['sensor_name'] == cis.name)
                         & (df['bg_density']  == bg)
                         & (df['operating_point'] == policy)].sort_values('velocity_px_s')
                if len(sub):
                    ax.plot(sub['velocity_px_s'], sub['power_mw'],
                            color=c, linestyle=ls, linewidth=2.2, alpha=alpha,
                            marker='o' if ls == '-' else 's', markersize=5)

        for dvs in DVS_SENSORS:
            c = DVS_COLORS[dvs.name]
            for theta, ls, lw, alpha in [
                (0.20, '-',  2.2, 1.0),
                (0.05, ':',  1.0, 0.5),
                (0.10, ':',  1.0, 0.5),
                (0.40, ':',  1.0, 0.5),
            ]:
                sub = df[(df['sensor_name'] == dvs.name)
                         & (df['bg_density']  == bg)
                         & (df['operating_point'] == f'theta={theta:.2f}')
                         ].sort_values('velocity_px_s')
                if len(sub):
                    ax.plot(sub['velocity_px_s'], sub['power_mw'],
                            color=c, linestyle=ls, linewidth=lw, alpha=alpha,
                            marker='^' if ls == '-' else None,
                            markersize=5)

        for cis in CIS_SENSORS:
            sub_cis = df[(df['sensor_name'] == cis.name)
                          & (df['bg_density']  == bg)
                          & (df['operating_point'] == 'adaptive')
                          ].sort_values('velocity_px_s')
            for dvs in DVS_SENSORS:
                sub_dvs = df[(df['sensor_name'] == dvs.name)
                              & (df['bg_density']  == bg)
                              & (df['operating_point'] == 'theta=0.20')
                              ].sort_values('velocity_px_s')
                if len(sub_cis) == 0 or len(sub_dvs) == 0:
                    continue
                v = sub_cis['velocity_px_s'].to_numpy()
                p_cis = sub_cis['power_mw'].to_numpy()
                p_dvs = sub_dvs['power_mw'].to_numpy()
                diff = p_cis - p_dvs
                sign = np.sign(diff)
                crossings = np.where(np.diff(sign) != 0)[0]
                for idx in crossings:
                    v0, v1 = v[idx], v[idx + 1]
                    d0, d1 = diff[idx], diff[idx + 1]
                    if d1 == d0:
                        continue
                    v_star = v0 - d0 * (v1 - v0) / (d1 - d0)
                    ax.axvline(v_star, color='gray', linestyle='-.',
                               linewidth=0.6, alpha=0.4)

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Velocity (px/s)')
        ax.set_ylabel('Total power (mW)')
        ax.grid(True, which='both', alpha=0.3)

    handles = []
    for c in CIS_SENSORS:
        handles.append(Line2D([], [], color=CIS_COLORS[c.name], linestyle='-',
                               marker='o', label=c.name))
    for d in DVS_SENSORS:
        handles.append(Line2D([], [], color=DVS_COLORS[d.name], linestyle='-',
                               marker='^', label=d.name))
    handles.append(Line2D([], [], color='gray', linestyle='-',  label='CIS adaptive'))
    handles.append(Line2D([], [], color='gray', linestyle='--', label='CIS locked'))
    handles.append(Line2D([], [], color='gray', linestyle=':',
                          label='DVS theta 0.05, 0.10, 0.40'))

    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.10, 1, 0.96])
    _save(fig, 'fig02_crossover_power_vs_velocity.png')


def _compute_vstar_table(df_a: pd.DataFrame, obj_size_px: int) -> pd.DataFrame:
    rows = []
    bg_values = sorted(df_a['bg_density'].unique())
    for cis in CIS_SENSORS:
        for dvs in DVS_SENSORS:
            for bg in bg_values:
                sub_cis = df_a[(df_a['sensor_name'] == cis.name)
                                & (df_a['obj_size_px'] == obj_size_px)
                                & (df_a['bg_density']  == bg)
                                & (df_a['operating_point'] == 'adaptive')
                                ].sort_values('velocity_px_s')
                sub_dvs = df_a[(df_a['sensor_name'] == dvs.name)
                                & (df_a['obj_size_px'] == obj_size_px)
                                & (df_a['bg_density']  == bg)
                                & (df_a['operating_point'] == 'theta=0.20')
                                ].sort_values('velocity_px_s')
                if len(sub_cis) == 0 or len(sub_dvs) == 0:
                    continue
                v = sub_cis['velocity_px_s'].to_numpy()
                diff = sub_cis['power_mw'].to_numpy() - sub_dvs['power_mw'].to_numpy()
                sign = np.sign(diff)
                idxs = np.where(np.diff(sign) != 0)[0]
                if len(idxs) == 0:
                    v_star = np.nan
                    dominant = 'DVS' if diff[0] > 0 else 'CIS'
                else:
                    idx = idxs[0]
                    v0, v1 = v[idx], v[idx + 1]
                    d0, d1 = diff[idx], diff[idx + 1]
                    v_star = v0 - d0 * (v1 - v0) / (d1 - d0) if d1 != d0 else v0
                    dominant = 'mixed'
                rows.append({
                    'cis_sensor': cis.name,
                    'dvs_sensor': dvs.name,
                    'bg_density': bg,
                    'v_star':     v_star,
                    'dominant':   dominant,
                })
    return pd.DataFrame(rows)


def fig03_vstar_vs_background(df_a: pd.DataFrame, obj_size_px: int = 50):
    """V star line plot as a continuous function of background density.

    One line per CIS DVS pair, one subplot per CIS sensor so the reader
    can see how each CIS degrades against each DVS candidate as the scene
    texture grows.
    """
    v_df = _compute_vstar_table(df_a, obj_size_px)
    v_df.to_csv(os.path.join(RESULTS, 'vstar_table.csv'), index=False)

    bg_values = sorted(v_df['bg_density'].unique())
    cis_names = sorted(v_df['cis_sensor'].unique())
    dvs_names = sorted(v_df['dvs_sensor'].unique())

    fig, axes = plt.subplots(1, len(cis_names),
                               figsize=(4.5 * len(cis_names), 6),
                               sharey=True)
    if len(cis_names) == 1:
        axes = [axes]
    fig.suptitle(f'V star crossover velocity vs background density (obj {obj_size_px}px)',
                 fontweight='bold')

    y_max = 0.0
    for ax, cis_name in zip(axes, cis_names):
        for dvs_name in dvs_names:
            sub = v_df[(v_df['cis_sensor'] == cis_name)
                       & (v_df['dvs_sensor'] == dvs_name)].sort_values('bg_density')
            if len(sub) == 0:
                continue
            finite_mask = sub['v_star'].notna()
            finite_sub  = sub[finite_mask]
            color = DVS_COLORS[dvs_name]
            if len(finite_sub):
                ax.plot(finite_sub['bg_density'], finite_sub['v_star'],
                         marker='o', color=color, linewidth=2.0,
                         label=dvs_name)
                y_max = max(y_max, finite_sub['v_star'].max())
            # Mark bg points where DVS dominates always
            miss_sub = sub[~finite_mask]
            for _, r in miss_sub.iterrows():
                ax.scatter(r['bg_density'], 0,
                            marker='x', color=color, s=60)

        ax.set_title(cis_name, fontsize=10)
        ax.set_xlabel('Background density d')
        ax.set_ylabel('V star (px/s), DVS wins above')
        ax.set_xticks(bg_values)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')

    for ax in axes:
        ax.set_ylim(bottom=-0.05 * max(y_max, 10))

    plt.tight_layout()
    _save(fig, 'fig03_vstar_vs_background.png')


def fig04_mot17_validation(df_b: pd.DataFrame):
    """MOT17 validation grid showing HOTA, DetA, AssA and MOTA per sensor type.

    HOTA is the top level score. DetA tells you whether the tracker
    actually found the objects. AssA tells you whether their IDs stayed
    consistent across frames. MOTA is the old single number summary that
    rolls everything into one. Showing all four next to each other makes
    it obvious why MOTA is misleading on MOT17.
    """
    if len(df_b) == 0:
        print('[F4] No Sweep B data, skipping')
        return
    df_b = df_b[df_b['source'] == 'mot17']
    agg = (df_b.groupby(
        ['sensor_name', 'sensor_type', 'operating_point', 'coast'])
           .agg(power_mw   = ('power_mw', 'mean'),
                hota_mean  = ('hota', 'mean'),
                hota_std   = ('hota', 'std'),
                det_a_mean = ('det_a', 'mean'),
                det_a_std  = ('det_a', 'std'),
                ass_a_mean = ('ass_a', 'mean'),
                ass_a_std  = ('ass_a', 'std'),
                mota_mean  = ('mota', 'mean'),
                mota_std   = ('mota', 'std'))
           .reset_index())
    for col in ['hota_std', 'det_a_std', 'ass_a_std', 'mota_std']:
        agg[col] = agg[col].fillna(0.0)

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('MOT17 validation: power vs tracking metrics, '
                  'HOTA broken into DetA and AssA',
                 fontweight='bold', y=0.995)

    def _plot(ax, subset, metric_mean, metric_std, ylabel, marker, color_map):
        for sensor_name, grp in subset.groupby('sensor_name'):
            c = color_map.get(sensor_name, 'gray')
            grp_sorted = grp.sort_values('power_mw')
            ax.errorbar(grp_sorted['power_mw'], grp_sorted[metric_mean],
                        yerr=grp_sorted[metric_std],
                        marker=marker, linestyle='-', color=c, linewidth=1.8,
                        capsize=4, markersize=10, label=sensor_name)
            for _, r in grp_sorted.iterrows():
                ax.annotate(r['operating_point'],
                             (r['power_mw'], r[metric_mean]),
                             textcoords='offset points', xytext=(6, 4),
                             fontsize=7)
        ax.set_xlabel('Power (mW)')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.legend(fontsize=8, loc='best')

    dvs_real = agg[(agg['sensor_type'] == 'DVS') & (~agg['coast'])]
    cis      = agg[agg['sensor_type'] == 'CIS']

    rows = [
        ('HOTA overall score',     'hota_mean',  'hota_std'),
        ('DetA, did you find them','det_a_mean', 'det_a_std'),
        ('AssA, did IDs stay',     'ass_a_mean', 'ass_a_std'),
        ('MOTA legacy metric',     'mota_mean',  'mota_std'),
    ]

    for i, (label, mean_col, std_col) in enumerate(rows):
        axes[i, 0].set_title(f'DVS  {label}', fontweight='bold')
        _plot(axes[i, 0], dvs_real, mean_col, std_col, label, '^', DVS_COLORS)

        axes[i, 1].set_title(f'CIS  {label}', fontweight='bold')
        _plot(axes[i, 1], cis, mean_col, std_col, label, 'o', CIS_COLORS)

        if mean_col == 'mota_mean':
            for col in (0, 1):
                axes[i, col].axhline(y=0, color='black',
                                      linewidth=0.8, linestyle=':')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    _save(fig, 'fig04_mot17_validation.png')


def fig05_coasting_comparison(df_b: pd.DataFrame):
    """Grouped bars comparing DVS HOTA with coasting on and off per threshold."""
    if len(df_b) == 0:
        print('[F5] No Sweep B data, skipping')
        return
    df_b = df_b[df_b['source'] == 'mot17']
    dvs = df_b[df_b['sensor_type'] == 'DVS'].copy()
    if len(dvs) == 0:
        print('[F5] No DVS rows, skipping')
        return

    agg = (dvs.groupby(['sensor_name', 'theta', 'coast'])
              .agg(hota_mean = ('hota', 'mean'),
                   hota_std  = ('hota', 'std'),
                   mota_mean = ('mota', 'mean'),
                   mota_std  = ('mota', 'std'))
              .reset_index())
    agg['hota_std'] = agg['hota_std'].fillna(0.0)
    agg['mota_std'] = agg['mota_std'].fillna(0.0)

    sensors = sorted(agg['sensor_name'].unique())
    fig, axes = plt.subplots(2, len(sensors), figsize=(5 * len(sensors), 10),
                               sharey='row')
    if len(sensors) == 1:
        axes = np.array(axes).reshape(2, 1)
    fig.suptitle('DVS coasting comparison: HOTA and MOTA with vs without coast',
                 fontweight='bold')

    thetas = sorted(agg['theta'].unique())
    x = np.arange(len(thetas))
    bar_w = 0.35

    for col, sensor_name in enumerate(sensors):
        sub = agg[agg['sensor_name'] == sensor_name]

        for row_idx, (metric_mean, metric_std, label) in enumerate([
            ('hota_mean', 'hota_std', 'HOTA'),
            ('mota_mean', 'mota_std', 'MOTA'),
        ]):
            ax = axes[row_idx, col]
            off_vals, off_stds, on_vals, on_stds = [], [], [], []
            for t in thetas:
                off = sub[(sub['theta'] == t) & (sub['coast'] == False)]
                on  = sub[(sub['theta'] == t) & (sub['coast'] == True)]
                off_vals.append(off[metric_mean].iloc[0] if len(off) else 0)
                off_stds.append(off[metric_std].iloc[0]  if len(off) else 0)
                on_vals.append(on[metric_mean].iloc[0]   if len(on)  else 0)
                on_stds.append(on[metric_std].iloc[0]    if len(on)  else 0)

            ax.bar(x - bar_w/2, off_vals, bar_w, yerr=off_stds,
                   color='#2255AA', label='coast off',
                   edgecolor='black', linewidth=0.8, capsize=4)
            ax.bar(x + bar_w/2, on_vals, bar_w, yerr=on_stds,
                   color='#cc4444', label='coast on',
                   edgecolor='black', linewidth=0.8, capsize=4)

            ax.set_xticks(x)
            ax.set_xticklabels([f'theta {t:.2f}' for t in thetas])
            if row_idx == 0:
                ax.set_title(sensor_name, fontsize=10)
            ax.set_ylabel(label)
            if row_idx == 1:
                ax.axhline(y=0, color='black', linewidth=0.8, linestyle=':')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, 'fig05_coasting_comparison.png')


def fig06_modulcis_calibration():
    """Scatter of linear CIS power against ModuCIS SPICE."""
    from modulcis_lut import ModulcisLut
    lut_path = os.path.join(RESULTS, 'modulcis_lut.json')
    if not os.path.exists(lut_path):
        print('[F6] No modulcis_lut.json, skipping')
        return
    lut = ModulcisLut.load(lut_path)

    rows = []
    for cis in CIS_SENSORS:
        for (w, h, fps_i, bits), spice_mw in lut.entries.items():
            if (int(w), int(h)) != cis.resolution or int(bits) != cis.adc_bits:
                continue
            linear_mw = cis.power_mw(float(fps_i))
            rows.append({
                'sensor':  cis.name,
                'fps':     int(fps_i),
                'spice':   float(spice_mw),
                'linear':  float(linear_mw),
            })

    if not rows:
        print('[F6] No matching CIS entries in LUT, skipping')
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle('ModuCIS SPICE vs linear interp calibration',
                 fontweight='bold')

    for sensor_name, grp in df.groupby('sensor'):
        ax.scatter(grp['linear'], grp['spice'],
                    s=80, alpha=0.85, edgecolor='black',
                    color=CIS_COLORS.get(sensor_name, 'gray'),
                    label=sensor_name)
        for _, r in grp.iterrows():
            ax.annotate(f'{r["fps"]}fps',
                         (r['linear'], r['spice']),
                         textcoords='offset points', xytext=(4, 4),
                         fontsize=7)

    lo = min(df['linear'].min(), df['spice'].min())
    hi = max(df['linear'].max(), df['spice'].max())
    ax.plot([lo, hi], [lo, hi], color='gray', linestyle='--',
             linewidth=1.0, label='y equals x')

    ax.set_xlabel('Linear interp power (mW) from sensor_database')
    ax.set_ylabel('ModuCIS SPICE power (mW) from CIS_Array')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, 'fig06_modulcis_calibration.png')


def fig07_mot17_vs_synthetic(df_b: pd.DataFrame):
    """Compare HOTA and MOTA on MOT17 real video against the synthetic scenes.

    MOT17 is hard, crowded, and the trackers plateau around the mid range
    on HOTA. The synthetic scenes are deliberately clean, so if the
    simulators behave reasonably we should see much higher scores on
    synthetic than on real video at every operating point, and the gap
    tells us how much of the tracker's headroom is blocked by the crowded
    MOT17 pedestrians rather than the sensor model.
    """
    if len(df_b) == 0:
        print('[F7] No Sweep B data, skipping')
        return
    have_sources = sorted(df_b['source'].dropna().unique().tolist())
    if 'mot17' not in have_sources or not any(s.startswith('synthetic') for s in have_sources):
        print('[F7] Missing MOT17 or synthetic rows, skipping')
        return

    # Drop coast=True rows to keep the comparison honest
    dfb = df_b[df_b['coast'] == False].copy()

    agg = (dfb.groupby(['sensor_name', 'sensor_type', 'operating_point', 'source'])
              .agg(power_mw  = ('power_mw', 'mean'),
                   hota_mean = ('hota', 'mean'),
                   hota_std  = ('hota', 'std'),
                   mota_mean = ('mota', 'mean'),
                   mota_std  = ('mota', 'std'))
              .reset_index())
    agg['hota_std'] = agg['hota_std'].fillna(0.0)
    agg['mota_std'] = agg['mota_std'].fillna(0.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('MOT17 real video vs synthetic scenes, HOTA and MOTA by source',
                 fontweight='bold')

    source_order  = ['mot17'] + sorted(
        s for s in have_sources if s.startswith('synthetic'))
    source_colors = {
        'mot17':          '#222222',
        'synthetic_low':  '#2ca02c',
        'synthetic_high': '#d62728',
    }
    source_markers = {
        'mot17':          'o',
        'synthetic_low':  's',
        'synthetic_high': 'D',
    }

    sensor_panels = [
        (axes[0, 0], 'DVS', 'hota_mean', 'hota_std', 'HOTA'),
        (axes[0, 1], 'CIS', 'hota_mean', 'hota_std', 'HOTA'),
        (axes[1, 0], 'DVS', 'mota_mean', 'mota_std', 'MOTA'),
        (axes[1, 1], 'CIS', 'mota_mean', 'mota_std', 'MOTA'),
    ]

    for ax, sensor_type, mean_col, std_col, metric_label in sensor_panels:
        sub = agg[agg['sensor_type'] == sensor_type]
        sensor_names = sorted(sub['sensor_name'].unique())
        x_bases = np.arange(len(sensor_names))
        bar_w = 0.22

        for i, source in enumerate(source_order):
            src_sub = sub[sub['source'] == source]
            vals = []
            errs = []
            for name in sensor_names:
                rows = src_sub[src_sub['sensor_name'] == name]
                vals.append(rows[mean_col].mean() if len(rows) else 0.0)
                errs.append(rows[std_col].mean()  if len(rows) else 0.0)
            offsets = x_bases + (i - (len(source_order) - 1) / 2) * bar_w
            ax.bar(offsets, vals, bar_w, yerr=errs,
                    color=source_colors.get(source, '#888888'),
                    edgecolor='black', linewidth=0.8, capsize=3,
                    label=source)

        ax.set_xticks(x_bases)
        ax.set_xticklabels(sensor_names, rotation=18, ha='right', fontsize=8)
        ax.set_ylabel(f'{metric_label} (mean across operating points and seeds)')
        ax.set_title(f'{sensor_type} {metric_label}', fontweight='bold')
        if metric_label == 'MOTA':
            ax.axhline(y=0, color='black', linewidth=0.8, linestyle=':')
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    _save(fig, 'fig07_mot17_vs_synthetic.png')


def write_design_rule_md(df_a: pd.DataFrame):
    """Write the design rule and V star table to results/design_rule.md."""
    v_df_path = os.path.join(RESULTS, 'vstar_table.csv')
    if not os.path.exists(v_df_path):
        fig03_vstar_vs_background(df_a)
    v_df = pd.read_csv(v_df_path)

    bg_low  = min(v_df['bg_density'].unique())
    bg_high = max(v_df['bg_density'].unique())

    def _pivot(bg):
        p = v_df[v_df['bg_density'] == bg].pivot(
            index='cis_sensor', columns='dvs_sensor', values='v_star')
        dom = v_df[v_df['bg_density'] == bg].pivot(
            index='cis_sensor', columns='dvs_sensor', values='dominant')
        return p, dom

    pl, dl = _pivot(bg_low)
    ph, dh = _pivot(bg_high)

    lines = []
    lines.append('# Design Rule')
    lines.append('')
    lines.append('DVS is preferred when object velocity exceeds V star and the')
    lines.append('background edge density stays below D max. CIS is preferred')
    lines.append('otherwise. The mechanism is that adaptive CIS power scales')
    lines.append('with the required FPS for a given velocity, while DVS power')
    lines.append('sits close to its static floor at every realistic event rate.')
    lines.append('Threshold theta has a negligible effect on total DVS power')
    lines.append('at 640 by 480 or higher.')
    lines.append('')
    lines.append(f'## V star table for low background (d {bg_low})')
    lines.append('')
    lines.append('| CIS sensor | ' + ' | '.join(pl.columns) + ' |')
    lines.append('|---|' + '|'.join(['---:'] * len(pl.columns)) + '|')
    for cis_name in pl.index:
        cells = [cis_name]
        for dvs_name in pl.columns:
            val = pl.loc[cis_name, dvs_name]
            dom = dl.loc[cis_name, dvs_name]
            if np.isnan(val):
                cells.append(f'{dom} always')
            else:
                cells.append(f'{val:.0f} px/s')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append(f'## V star table for high background (d {bg_high})')
    lines.append('')
    lines.append('| CIS sensor | ' + ' | '.join(ph.columns) + ' |')
    lines.append('|---|' + '|'.join(['---:'] * len(ph.columns)) + '|')
    for cis_name in ph.index:
        cells = [cis_name]
        for dvs_name in ph.columns:
            val = ph.loc[cis_name, dvs_name]
            dom = dh.loc[cis_name, dvs_name]
            if np.isnan(val):
                cells.append(f'{dom} always')
            else:
                cells.append(f'{val:.0f} px/s')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')

    out = os.path.join(RESULTS, 'design_rule.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'[Saved] {out}')


def main():
    df_a = _load_sweep_a()
    df_b = _load_sweep_b()

    print(f'[Figures] Sweep A rows: {len(df_a)}')
    print(f'[Figures] Sweep B rows: {len(df_b)}')

    fig02_crossover(df_a)
    fig03_vstar_vs_background(df_a)
    fig04_mot17_validation(df_b)
    fig05_coasting_comparison(df_b)
    fig06_modulcis_calibration()
    fig07_mot17_vs_synthetic(df_b)
    write_design_rule_md(df_a)
    print('\n[Figures] done')


if __name__ == '__main__':
    main()
