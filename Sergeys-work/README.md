# CIS vs DVS Power Crossover

Finds v\*, the velocity where DVS becomes cheaper than CIS. Runs an
analytical sweep across 4 DVS + 4 CIS sensors, then validates with
tracking on MOT17-04-SDP + synthetic scenes.

## Run it

```bash
pip install -r requirements.txt

python -m models.cis_spice_lut
python run_sweeps.py --max-frames 1050 --num-seeds 5 --workers 16
python generate_figures.py
```

Sweep A: ~1 s. Sweep B: ~8 min at 16 workers. (on i7-14700k).
Outputs → `results/` and `results/figures/`.

## Layout

```
run_sweeps.py         # Sweep A + B
generate_figures.py   # F1..F7

data/
  mot17_loader.py        # MOT17 ingest
  sensor_simulators.py   # CIS/DVS noisy detections + synthetic scenes

models/
  sensor_database.py     # datasheet specs, 4 DVS + 4 CIS
  power_crossover.py     # analytical power formulas + Sweep A
  cis_spice_lut.py       # cached SPICE (F6 calibration only)

evaluation/
  sort_tracker.py        # SORT (Kalman + Hungarian)
  tracking_metrics.py    # HOTA / DetA / AssA / MOTA / IDF1
```

## Sweeps

**Sweep A** — 7 velocities × 3 sizes × 5 bg
densities × (4 CIS × 2 policies + 4 DVS × 4 thresholds). Ramaa's
`compute_event_rate` → `event_rate_at_theta` (1/theta scaling) → Ish's
circuit formula for DVS, datasheet interp for CIS.

**Sweep B** — Same sensor grid → simulators → SORT →
HOTA/MOTA. Sources: MOT17-04-SDP (1050 frames, 5 seeds, coast on+off =
240 runs) and two synthetic scenes (300 frames, 3 seeds = 192 runs).

## Figures

- **F1** pipeline diagram (`results/fig01_pipeline_architecture.md`)
- **F2** power vs velocity with v\* stars, low + high bg
- **F3** v\* vs bg density, one panel per CIS
- **F4** HOTA/DetA/AssA/MOTA vs power on MOT17
- **F5** DVS coast on vs off
- **F6** linear CIS power vs SPICE
- **F7** MOT17 vs synthetic

## Caveats

- Real video is MOT17-04-SDP only. Full split would add 5 GB.
- SPICE runs ~2× datasheet (generic 65 nm). Sweeps use datasheets,
  SPICE only feeds F6.
- DVS HOTA plateaus ~0.35 — simulator drops events below 50/frame,
  which hurts slow objects.
- Position noise scales as 1920/sensor_width, so smaller sensors look
  worse than on native-res video.
- FP rates default to 2.0 DVS / 2.5 CIS per frame.
- MOTA is negative-normal on MOT17-04-SDP with these weak detectors.
  Compare deltas, not absolutes.
