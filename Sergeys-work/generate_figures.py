"""Reads the sweep CSVs and writes the seven slide figures to results/figures/.

Run after run_sweeps.py. F1 is the pipeline architecture diagram, F2 is
the main power vs velocity crossover plot with both CIS policies and all
four DVS thresholds, F3 and F4 give the V star breakdown and the design
rule table, F5 shows HOTA on real MOT17 video, F6 is the coasting on vs
off comparison, and F7 checks the CIS linear power model against the
ModuCIS SPICE values. All figures share one colour scheme, blues for CIS
and reds for DVS, written at 300 DPI.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def fig01_pipeline_architecture():
    """Boxes and arrows diagram of how the sweep pipeline fits together."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')
    ax.set_title('Unified CIS vs DVS Power-Tracking Pipeline',
                 fontsize=16, fontweight='bold', pad=10)

    def box(x, y, w, h, text, fc='#DDEEFF'):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.1',
            facecolor=fc, edgecolor='#333', linewidth=1.5))
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=5)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#444'))

    box(0.5, 6.0, 3.5, 2.0,
        "Ramaa's scene model\n(velocity, obj_size,\nbackground density)",
        fc='#FFE8CC')
    box(5.0, 6.5, 3.5, 1.2,
        "Ramaa bridges\n• compute_event_rate (+θ fix)\n• compute_fps_min",
        fc='#FFF3CD')
    box(5.0, 4.8, 3.5, 1.2,
        'sensor_database.py\n8 sensors (DVS + CIS)\nper-sensor datasheet',
        fc='#D6E6FF')
    box(9.5, 6.5, 3.0, 1.2,
        "Harshitha's ModuCIS\n(SPICE, LUT cached)",
        fc='#D6FFD6')
    box(9.5, 4.8, 3.0, 1.2,
        "Ish's DVS circuit\n(θ² energy, FP/FN)",
        fc='#FFD6D6')
    box(13.2, 5.7, 2.5, 1.0,
        'unified_crossover.py\npower formulas',
        fc='#E0D4F7')
    box(5.5, 2.3, 4.5, 1.5,
        'Sweep A\nclosed-form, 1008 rows\n→ power vs velocity',
        fc='#CCE5FF')
    box(10.5, 2.3, 4.5, 1.5,
        'Sweep B\nMOT17-04-SDP pixels\nHOTA/DetA/AssA + coast ablation',
        fc='#FFCCCC')
    box(8.0, 0.3, 5.0, 1.3,
        '7 slide-ready figures\nF1..F7, 300 DPI',
        fc='#FFE5B4')

    arrow(2.25, 6.0, 5.0, 7.0)
    arrow(2.25, 6.0, 5.0, 5.4)
    arrow(8.5, 7.1, 9.5, 7.1)
    arrow(8.5, 5.4, 9.5, 5.4)
    arrow(12.5, 7.1, 13.2, 6.5)
    arrow(12.5, 5.4, 13.2, 6.0)
    arrow(14.4, 5.7, 8.0, 3.8)
    arrow(14.4, 5.7, 13.0, 3.8)
    arrow(7.75, 2.3, 10.5, 1.6)
    arrow(12.75, 2.3, 10.5, 1.6)

    _save(fig, 'fig01_pipeline_architecture.png')


def fig02_crossover(df_a: pd.DataFrame, obj_size_px: int = 50):
    """Main power vs velocity plot with CIS and DVS curves and crossover marks."""
    df = df_a[df_a['obj_size_px'] == obj_size_px].copy()
    bg_vals = sorted(df['bg_density'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle(f'CIS vs DVS Power-Velocity Crossover (obj={obj_size_px}px)',
                 fontweight='bold')

    for ax, bg in zip(axes, bg_vals):
        bg_label = df[df['bg_density'] == bg]['bg_label'].iloc[0]
        ax.set_title(f'{bg_label} (d={bg})', fontweight='bold')

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

        # Crossover markers
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
    handles.append(Line2D([], [], color='gray', linestyle=':',  label='DVS θ={0.05, 0.10, 0.40}'))

    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.10, 1, 0.96])
    _save(fig, 'fig02_crossover_power_vs_velocity.png')


def fig03_vstar_vs_background(df_a: pd.DataFrame, obj_size_px: int = 50):
    """Bar chart of V star for each CIS DVS pair at low and high texture."""
    bg_values = sorted(df_a['bg_density'].unique())
    rows = []
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
                    dominant = '—'
                rows.append({
                    'cis_sensor': cis.name,
                    'dvs_sensor': dvs.name,
                    'bg_density': bg,
                    'v_star':     v_star,
                    'dominant':   dominant,
                })

    v_df = pd.DataFrame(rows)
    v_df.to_csv(os.path.join(RESULTS, 'vstar_table.csv'), index=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(f'V* Crossover Velocity vs Background Density (obj={obj_size_px}px)',
                 fontweight='bold')

    cis_names = sorted(v_df['cis_sensor'].unique())
    dvs_names = sorted(v_df['dvs_sensor'].unique())
    x_bases = np.arange(len(cis_names))
    bar_width = 0.10

    for i, dvs_name in enumerate(dvs_names):
        for j, bg in enumerate(bg_values):
            sub = v_df[(v_df['dvs_sensor'] == dvs_name)
                       & (v_df['bg_density']  == bg)]
            vals = [
                sub[sub['cis_sensor'] == c]['v_star'].iloc[0]
                if len(sub[sub['cis_sensor'] == c]) else np.nan
                for c in cis_names
            ]
            # Replace NaN with a small sentinel for plotting (will be labelled)
            plot_vals = [v if not np.isnan(v) else 0 for v in vals]
            offsets = x_bases + (i * 2 + j - 3.5) * bar_width
            label = f'{dvs_name} ({["low","high"][j]})' if True else None
            ax.bar(offsets, plot_vals, bar_width,
                    color=DVS_COLORS[dvs_name],
                    alpha=0.85 if j == 0 else 0.45,
                    edgecolor='black', linewidth=0.6,
                    label=label)
            for off, val in zip(offsets, vals):
                if np.isnan(val):
                    ax.text(off, 5, 'DVS\nalways', ha='center',
                             va='bottom', fontsize=6, color='darkred')
                else:
                    ax.text(off, val + 3, f'{val:.0f}',
                             ha='center', va='bottom', fontsize=6, rotation=70)

    ax.set_xticks(x_bases)
    ax.set_xticklabels(cis_names, rotation=15, ha='right')
    ax.set_ylabel('V* (px/s) — DVS wins above this velocity')
    ax.set_title('Per (CIS, DVS) pair; bold bars = low texture, light bars = high texture')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    plt.tight_layout()
    _save(fig, 'fig03_vstar_vs_background.png')


def fig04_design_rule(df_a: pd.DataFrame):
    """Title card with the design rule statement plus a V star table."""
    v_df_path = os.path.join(RESULTS, 'vstar_table.csv')
    if not os.path.exists(v_df_path):
        fig03_vstar_vs_background(df_a)
    v_df = pd.read_csv(v_df_path)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Design Rule — When Does DVS Beat CIS?',
                 fontsize=16, fontweight='bold')

    ax_text = fig.add_axes([0.06, 0.62, 0.88, 0.30])
    ax_text.axis('off')
    ax_text.text(0.5, 0.75,
                 'DVS preferred when:   velocity > V*   AND   bg density < D_max',
                 ha='center', va='center', fontsize=18, fontweight='bold')
    ax_text.text(0.5, 0.3,
                 'CIS preferred otherwise.\n'
                 'Mechanism: adaptive CIS power scales with required FPS.\n'
                 'DVS power is near-constant at its static floor for realistic event rates.\n'
                 'Threshold θ has ~zero effect on total DVS power at 640×480 or higher.',
                 ha='center', va='center', fontsize=11, style='italic',
                 color='#555')

    ax_table = fig.add_axes([0.06, 0.05, 0.88, 0.48])
    ax_table.axis('off')

    bg_low  = min(v_df['bg_density'].unique())
    bg_high = max(v_df['bg_density'].unique())
    pivot_low  = v_df[v_df['bg_density'] == bg_low] .pivot(
        index='cis_sensor', columns='dvs_sensor', values='v_star')
    pivot_high = v_df[v_df['bg_density'] == bg_high].pivot(
        index='cis_sensor', columns='dvs_sensor', values='v_star')
    pivot_dom_low = v_df[v_df['bg_density'] == bg_low].pivot(
        index='cis_sensor', columns='dvs_sensor', values='dominant')

    cell_text = []
    for cis_name in pivot_low.index:
        row = [cis_name]
        for dvs_name in pivot_low.columns:
            lo = pivot_low.loc[cis_name, dvs_name]
            hi = pivot_high.loc[cis_name, dvs_name] if dvs_name in pivot_high.columns else np.nan
            dom = pivot_dom_low.loc[cis_name, dvs_name] if dvs_name in pivot_dom_low.columns else '—'
            if np.isnan(lo):
                lo_s = f'{dom} always'
            else:
                lo_s = f'{lo:.0f}'
            if np.isnan(hi):
                hi_s = '—'
            else:
                hi_s = f'{hi:.0f}'
            row.append(f'low: {lo_s}\nhigh: {hi_s}')
        cell_text.append(row)

    col_labels = ['CIS \\ DVS'] + list(pivot_low.columns)
    table = ax_table.table(
        cellText=cell_text, colLabels=col_labels,
        cellLoc='center', loc='center',
        colColours=['#2255AA'] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 2.4)
    for j in range(len(col_labels)):
        table[0, j].get_text().set_color('white')
        table[0, j].get_text().set_fontweight('bold')

    ax_table.set_title(f'V* (px/s) per sensor pair  —  low texture (d={bg_low}) / high texture (d={bg_high})',
                       fontsize=11, fontweight='bold', pad=15)

    _save(fig, 'fig04_design_rule.png')


def fig05_mot17_validation(df_b: pd.DataFrame):
    """Power vs HOTA from the MOT17 tracking runs, one panel per sensor type."""
    if len(df_b) == 0:
        print('[F5] No Sweep B data — skipping')
        return
    agg = (df_b.groupby(
        ['sensor_name', 'sensor_type', 'operating_point', 'coast'])
           .agg(power_mw=('power_mw', 'mean'),
                hota_mean=('hota', 'mean'),
                hota_std =('hota', 'std'))
           .reset_index())
    agg['hota_std'] = agg['hota_std'].fillna(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('MOT17-04-SDP Validation — Power vs HOTA',
                 fontweight='bold')

    ax = axes[0]
    ax.set_title('DVS (coasting off — the real performance)', fontweight='bold')
    dvs = agg[(agg['sensor_type'] == 'DVS') & (~agg['coast'])]
    for sensor_name, grp in dvs.groupby('sensor_name'):
        c = DVS_COLORS.get(sensor_name, 'gray')
        grp_sorted = grp.sort_values('power_mw')
        ax.errorbar(grp_sorted['power_mw'], grp_sorted['hota_mean'],
                    yerr=grp_sorted['hota_std'],
                    marker='^', linestyle='-', color=c, linewidth=1.8,
                    capsize=4, markersize=10, label=sensor_name)
        for _, r in grp_sorted.iterrows():
            ax.annotate(r['operating_point'], (r['power_mw'], r['hota_mean']),
                         textcoords='offset points', xytext=(6, 4),
                         fontsize=7)
    ax.set_xlabel('Power (mW)'); ax.set_ylabel('HOTA (mean ± std)')
    ax.set_xscale('log'); ax.legend(fontsize=8, loc='best')

    ax = axes[1]
    ax.set_title('CIS (FPS sweep)', fontweight='bold')
    cis = agg[agg['sensor_type'] == 'CIS']
    for sensor_name, grp in cis.groupby('sensor_name'):
        c = CIS_COLORS.get(sensor_name, 'gray')
        grp_sorted = grp.sort_values('power_mw')
        ax.errorbar(grp_sorted['power_mw'], grp_sorted['hota_mean'],
                    yerr=grp_sorted['hota_std'],
                    marker='o', linestyle='-', color=c, linewidth=1.8,
                    capsize=4, markersize=10, label=sensor_name)
        for _, r in grp_sorted.iterrows():
            ax.annotate(r['operating_point'], (r['power_mw'], r['hota_mean']),
                         textcoords='offset points', xytext=(6, 4),
                         fontsize=7)
    ax.set_xlabel('Power (mW)'); ax.set_ylabel('HOTA (mean ± std)')
    ax.set_xscale('log'); ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    _save(fig, 'fig05_mot17_validation_hota.png')


def fig06_coasting_ablation(df_b: pd.DataFrame):
    """Grouped bars comparing DVS HOTA with coasting on and off per threshold."""
    if len(df_b) == 0:
        print('[F6] No Sweep B data — skipping')
        return
    dvs = df_b[df_b['sensor_type'] == 'DVS'].copy()
    if len(dvs) == 0:
        print('[F6] No DVS rows — skipping')
        return

    agg = (dvs.groupby(['sensor_name', 'theta', 'coast'])
              .agg(hota_mean=('hota', 'mean'),
                   hota_std =('hota', 'std'))
              .reset_index())
    agg['hota_std'] = agg['hota_std'].fillna(0.0)

    sensors = sorted(agg['sensor_name'].unique())
    fig, axes = plt.subplots(1, len(sensors), figsize=(5 * len(sensors), 6),
                               sharey=True)
    if len(sensors) == 1:
        axes = [axes]
    fig.suptitle('DVS Coasting: HOTA with vs without coast',
                 fontweight='bold')

    thetas = sorted(agg['theta'].unique())
    x = np.arange(len(thetas))
    bar_w = 0.35

    for ax, sensor_name in zip(axes, sensors):
        sub = agg[agg['sensor_name'] == sensor_name]

        off_vals = []
        off_stds = []
        on_vals  = []
        on_stds  = []
        for t in thetas:
            off = sub[(sub['theta'] == t) & (sub['coast'] == False)]
            on  = sub[(sub['theta'] == t) & (sub['coast'] == True)]
            off_vals.append(off['hota_mean'].iloc[0] if len(off) else 0)
            off_stds.append(off['hota_std'].iloc[0]  if len(off) else 0)
            on_vals.append(on['hota_mean'].iloc[0]   if len(on)  else 0)
            on_stds.append(on['hota_std'].iloc[0]    if len(on)  else 0)

        ax.bar(x - bar_w/2, off_vals, bar_w, yerr=off_stds,
               color='#2255AA', label='coast OFF (real)',
               edgecolor='black', linewidth=0.8, capsize=4)
        ax.bar(x + bar_w/2, on_vals, bar_w, yerr=on_stds,
               color='#cc4444', label='coast ON (legacy cheat)',
               edgecolor='black', linewidth=0.8, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels([f'θ={t:.2f}' for t in thetas])
        ax.set_title(sensor_name, fontsize=10)
        ax.set_ylabel('HOTA')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, 'fig06_coasting_comparison.png')


def fig07_modulcis_calibration():
    """Scatter of linear CIS power against ModuCIS SPICE, with a y equals x line."""
    from modulcis_lut import ModulcisLut
    lut_path = os.path.join(RESULTS, 'modulcis_lut.json')
    if not os.path.exists(lut_path):
        print('[F7] No modulcis_lut.json — skipping')
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
        print('[F7] No matching CIS entries in LUT — skipping')
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle('ModuCIS SPICE vs Linear-Interp Calibration',
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
             linewidth=1.0, label='y = x')

    ax.set_xlabel('Linear-interp power (mW)  —  sensor.power_mw(fps)')
    ax.set_ylabel('ModuCIS SPICE power (mW)  —  CIS_Array.system_total_power')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, 'fig07_modulcis_calibration.png')


def main():
    df_a = _load_sweep_a()
    df_b = _load_sweep_b()

    print(f'[Figures] Sweep A rows: {len(df_a)}')
    print(f'[Figures] Sweep B rows: {len(df_b)}')

    fig01_pipeline_architecture()
    fig02_crossover(df_a)
    fig03_vstar_vs_background(df_a)
    fig04_design_rule(df_a)
    fig05_mot17_validation(df_b)
    fig06_coasting_ablation(df_b)
    fig07_modulcis_calibration()
    print('\n[Figures] all 7 generated')


if __name__ == '__main__':
    main()
