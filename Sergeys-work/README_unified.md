# Unified CIS vs DVS Crossover Pipeline

This directory is Team 4's Gap 3 closure. It wires Ramaa's scene model to
Harshitha's ModuCIS power numbers and Ish's DVS circuit formula so both
sensor types can be compared on the same axes, then runs everything on
real MOT17 video to check the story holds up.

The headline result is that DVS becomes cheaper than CIS once object
velocity crosses some V star that depends on sensor resolution and object
size. V star is the velocity at which the CIS required FPS drives its
power above the DVS static floor. The crossover is really a CIS story,
since DVS total power stays close to its static floor at every event rate
we care about at 640 by 480 or higher.

## Headline numbers

From `results/vstar_table.csv`, for a 50 pixel object against a low texture
background. Values are the crossover velocity in px/s.

| CIS | vs Samsung DVS-Gen3.1 | vs Prophesee IMX636 |
|---|---:|---:|
| OV7251 (VGA)   | 126           | 474           |
| IMX327 (1080p) | DVS always    | 18            |
| AR0234 (1200p) | DVS always    | 55            |
| IMX462 (1080p) | DVS always    | 40            |

`results/sweep_b_mot17.csv` also shows how much the DVS coasting trick
inflates HOTA. Samsung Gen3.1 picks up 0.079, DAVIS346 picks up 0.019,
Prophesee picks up 0.014 and Lichtsteiner picks up 0.011. The legacy
simulator was quietly applying this on every run.

## How to reproduce

From this directory, run the ModuCIS LUT builder once if it is not already
cached. It takes about three minutes because it instantiates Harshitha's
SPICE model once per operating point.

```bash
python modulcis_lut.py
```

Then run both sweeps. Sweep A is closed form and finishes in about a
second. Sweep B runs 240 tracking configurations through the simulators
and HOTA metrics on 16 workers, which takes around nine minutes for the
full 1050 frames at 5 seeds.

```bash
python run_sweeps.py --max-frames 1050 --num-seeds 5 --workers 16
python generate_figures.py
```

Worker count of 28 triggered an IMX462 out of memory on a 32 GB Windows
box during testing, so 16 is the safer default. Outputs land in `results/`
and `results/figures/`.

## Files

`unified_crossover.py` holds the scene to event bridge, the analytical
CIS and DVS power functions and the Sweep A driver. `noise_models.py` has
the CIS and DVS trajectory simulators with the explicit coasting flag and
Poisson false positive injection. `metrics_hota.py` computes HOTA, DetA
and AssA directly from a motmetrics accumulator. `modulcis_lut.py` caches
Harshitha's SPICE model output to JSON for the F7 calibration figure.
`run_sweeps.py` is the master driver that calls Sweep A and Sweep B.
`generate_figures.py` reads both sweep CSVs and writes the seven figures.
`tests/` holds 18 unit tests covering HOTA, the power formulas and the
simulators.

The rest of Sergeys-work is shared infrastructure that predates this
rebuild and is not touched: `ingest_mot.py`, `fast_sort.py`, `fast_eval.py`,
`sort_tracker.py`, `simple_trackers.py`, `cis_detector.py`, `event_frames.py`,
`sensor_database.py`, `v2e_adapter.py`, `run_benchmark.py` and
`evaluate_tracking.py`.

## Sweep A and Sweep B

Sweep A is the closed form analytical crossover. It walks 7 velocities by
3 object sizes by 2 background densities, and for each of those it runs
4 CIS sensors under 2 policies plus 4 DVS sensors at 4 thresholds. That
is 1008 rows total. Ramaa's `compute_event_rate` gives the base event
rate, `unified_crossover.event_rate_at_theta` scales that by 1 over theta
to close the Gap 2 threshold coupling, `dvs_power_custom` applies Ish's
circuit formula with per sensor datasheet constants, and `cis_power_custom`
uses the locked or adaptive FPS policy to pick a frame rate and read the
matching power from `sensor_database`.

Sweep B is the MOT17 validation. It runs each of 4 DVS sensors at 4
thresholds, both with and without coasting, at 5 seeds, plus 4 CIS sensors
at 4 FPS settings at 5 seeds. That is 240 tracking runs. Each one takes
the MOT17-04-SDP ground truth through the matching simulator from
`noise_models`, pushes the noisy detections through the SORT tracker and
evaluates with HOTA, DetA, AssA, MOTA, IDF1 and ID switches. Power for
each run comes from the same analytical model Sweep A uses, so the two
sweeps are directly comparable.

## Figures

`fig01` is the pipeline architecture diagram. `fig02` is the main power
versus velocity crossover plot with both CIS policies and all DVS
thresholds, with crossover markers where the adaptive CIS curves meet the
DVS bands. `fig03` shows V star per CIS DVS pair at low and high texture.
`fig04` is the design rule title card plus the V star table. `fig05`
shows power versus HOTA on real MOT17 video for both sensor types.
`fig06_coasting_comparison` exposes the hidden DVS uplift from coasting.
`fig07` compares the linear CIS power approximation against ModuCIS SPICE.

## Caveats

Everything runs on MOT17-04-SDP only. Reporting the full MOT17 training
split would need another 5 GB of data and was skipped to keep iteration
fast. ModuCIS SPICE tends to run about two times higher than the
sensor_database datasheet numbers because it uses a generic 65 nm process
model, so the sweep uses the datasheet values and ModuCIS only feeds F7.
The DVS tracking HOTA plateaus around 0.35 because the simulator drops
events below 50 per frame, which hurts slow objects. The position noise
also scales with 1920 over sensor width, so smaller sensors look worse
than they would on native resolution video. False positive rates default
to 2.0 for DVS and 2.5 for CIS per frame; a proper calibration against
real MOG2 plus SORT output would be a simple follow up if the F5 numbers
need tightening.
