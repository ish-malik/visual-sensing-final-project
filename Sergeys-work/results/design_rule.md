# Design Rule

DVS is preferred when object velocity exceeds V star and the
background edge density stays below D max. CIS is preferred
otherwise. The mechanism is that adaptive CIS power scales
with the required FPS for a given velocity, while DVS power
sits close to its static floor at every realistic event rate.
Threshold theta has a negligible effect on total DVS power
at 640 by 480 or higher.

## V star table for low background (d 0.05)

| CIS sensor | DAVIS346 | Lichtsteiner 2008 | Prophesee IMX636 | Samsung DVS-Gen3.1 |
|---|---:|---:|---:|---:|
| AR0234 (ON Semi) | DVS always | 23 px/s | 55 px/s | DVS always |
| IMX327 (Sony) | DVS always | DVS always | 18 px/s | DVS always |
| IMX462 (Sony) | DVS always | 14 px/s | 40 px/s | DVS always |
| OV7251 (OmniVision) | 127 px/s | 280 px/s | 474 px/s | 126 px/s |

## V star table for high background (d 0.4)

| CIS sensor | DAVIS346 | Lichtsteiner 2008 | Prophesee IMX636 | Samsung DVS-Gen3.1 |
|---|---:|---:|---:|---:|
| AR0234 (ON Semi) | DVS always | 23 px/s | 55 px/s | DVS always |
| IMX327 (Sony) | DVS always | DVS always | 18 px/s | DVS always |
| IMX462 (Sony) | DVS always | 15 px/s | 40 px/s | DVS always |
| OV7251 (OmniVision) | 128 px/s | 311 px/s | 475 px/s | 127 px/s |
