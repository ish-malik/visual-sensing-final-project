# Unified CIS vs DVS Crossover Pipeline

This directory is Team 4's Gap 3 closure. It wires Ramaa's scene model
to Harshitha's ModuCIS power numbers and Ish's DVS circuit formula so
both sensor types can be compared on the same axes, then runs everything
on real MOT17 video plus a synthetic scene generator to see whether the
story holds up when the tracker has to work with real pixels.

The headline result is that DVS becomes cheaper than CIS once object
velocity crosses some V star that depends on sensor resolution and
object size. V star is the velocity at which the CIS required FPS drives
its power above the DVS static floor. The crossover is really a CIS
story, since DVS total power stays close to its static floor at every
event rate we care about at 640 by 480 or higher.

## Headline numbers

From `results/design_rule.md`, for a 50 pixel object. The full V star
table across low and high background density is in that file, but the
headline pairs at low density are:

| CIS | vs Samsung DVS-Gen3.1 | vs Prophesee IMX636 |
|---|---:|---:|
| OV7251 (VGA)   | 126           | 474           |
| IMX327 (1080p) | DVS always    | 18            |
| AR0234 (1200p) | DVS always    | 55            |
| IMX462 (1080p) | DVS always    | 40            |

`results/sweep_b_mot17.csv` also shows the DVS coasting uplift. Samsung
Gen3.1 picks up 0.079 HOTA, DAVIS346 picks up 0.019, Prophesee picks up
0.014 and Lichtsteiner picks up 0.011. The legacy simulator was quietly
applying this on every run.

## How to reproduce

From this directory, run the ModuCIS LUT builder once if it is not
already cached. It takes about three minutes because it instantiates
Harshitha's SPICE model once per operating point.

```bash
python modulcis_lut.py
```

Then run both sweeps. Sweep A is closed form and finishes in about a
second. Sweep B runs 432 tracking configurations through the simulators
and HOTA metrics on 16 workers, which takes around eight minutes for
1050 MOT17 frames plus two 300 frame synthetic scenes.

```bash
python run_sweeps.py --max-frames 1050 --num-seeds 5 --workers 16
python generate_figures.py
```

Worker count of 28 triggered an IMX462 out of memory on a 32 GB Windows
box during testing, so 16 is the safer default. Outputs land in
`results/` and `results/figures/`.

## Files

`unified_crossover.py` holds the scene to event bridge, the analytical
CIS and DVS power functions and the Sweep A driver. `noise_models.py`
has the CIS and DVS trajectory simulators with the explicit coasting
flag, the Poisson false positive injection and the synthetic scene
generator. `metrics_hota.py` computes HOTA, DetA and AssA directly from
a motmetrics accumulator. `modulcis_lut.py` caches Harshitha's SPICE
model output to JSON for the F6 calibration figure. `run_sweeps.py` is
the master driver that calls Sweep A and Sweep B with both the MOT17
and the synthetic sources. `generate_figures.py` reads both sweep CSVs
and writes the figures. `tests/` holds 18 unit tests covering HOTA, the
power formulas and the simulators.

The rest of Sergeys-work is shared infrastructure that predates this
rebuild and is not touched: `ingest_mot.py`, `fast_sort.py`,
`sort_tracker.py`, `simple_trackers.py`, `cis_detector.py`,
`event_frames.py`, `sensor_database.py`, `v2e_adapter.py`,
`run_benchmark.py` and `evaluate_tracking.py`.

## Sweep A and Sweep B

Sweep A is the closed form analytical crossover. It walks 7 velocities
by 3 object sizes by 5 background densities, and for each of those it
runs 4 CIS sensors under 2 policies plus 4 DVS sensors at 4 thresholds.
That is 2520 rows total. Ramaa's `compute_event_rate` gives the base
event rate, `unified_crossover.event_rate_at_theta` scales that by 1
over theta to close the Gap 2 threshold coupling, `dvs_power_custom`
applies Ish's circuit formula with per sensor datasheet constants, and
`cis_power_custom` uses the locked or adaptive FPS policy to pick a
frame rate and read the matching power from `sensor_database`.

Sweep B is the tracking validation. It runs a grid of 4 DVS sensors at
4 thresholds plus 4 CIS sensors at 4 FPS settings across two sources:
MOT17-04-SDP real video (1050 frames, 5 seeds, coasting on and off) and
two synthetic scenes (300 frames, 3 seeds, low and high background
density, 200 px per second objects). That is 240 MOT17 runs plus 192
synthetic runs, so 432 total. Each run takes the matching ground truth
through the simulator in `noise_models`, pushes the noisy detections
through the SORT tracker and evaluates with HOTA, DetA, AssA, MOTA,
IDF1 and ID switches. Power for each run comes from the same analytical
model Sweep A uses, so the two sweeps are directly comparable.

## Figures

F1 is the pipeline architecture Mermaid diagram in
`results/fig01_pipeline_architecture.md`, which renders in GitHub and
VS Code preview. F2 is the main power versus velocity crossover plot
with both CIS policies and all DVS thresholds, with crossover markers
where the adaptive CIS curves meet the DVS bands, shown at low and high
background density. F3 is the V star line chart across the five bg
density values with one panel per CIS sensor. F4 is the MOT17 power
versus tracking metrics grid showing HOTA, DetA, AssA and MOTA so the
HOTA split into detection quality and association quality is visible.
F5 is the coast on versus coast off comparison for DVS across HOTA and
MOTA. F6 compares the linear CIS power approximation against ModuCIS
SPICE. F7 compares MOT17 results against the synthetic scene runs so
we can see how much harder real pedestrian video is than the synthetic
model assumes. The design rule table lives in `results/design_rule.md`.

## Caveats

Everything runs on MOT17-04-SDP only for the real video half. Reporting
the full MOT17 training split would need another 5 GB of data and was
skipped to keep iteration fast. ModuCIS SPICE tends to run about two
times higher than the sensor_database datasheet numbers because it uses
a generic 65 nm process model, so the sweep uses the datasheet values
and ModuCIS only feeds F6. The DVS tracking HOTA plateaus around 0.35
because the simulator drops events below 50 per frame, which hurts slow
objects. The position noise also scales with 1920 over sensor width, so
smaller sensors look worse than they would on native resolution video.
False positive rates default to 2.0 for DVS and 2.5 for CIS per frame;
a proper calibration against real MOG2 plus SORT output would be a
simple follow up if the F4 or F7 numbers need tightening.
