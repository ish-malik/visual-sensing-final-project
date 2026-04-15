"""One shot ModuCIS power lookup table.

Harshitha's CIS_Array SPICE model is slow to instantiate, so we run it
once per resolution, FPS and ADC bit combination and cache the total
power to modulcis_lut.json. The main sweep uses the faster linear
interpolation from sensor_database instead. This LUT only feeds the F7
calibration figure that shows how close that linear approximation is to
the real ModuCIS numbers.
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE       = os.path.dirname(os.path.abspath(__file__))
PROJECT    = os.path.dirname(HERE)
CIS_MODEL  = os.path.join(PROJECT, 'Harshithas-work', 'cis_model')
USECASES   = os.path.join(CIS_MODEL, 'usecases')
COMPONENTS = os.path.join(CIS_MODEL, 'model_components')

sys.path.insert(0, USECASES)
sys.path.insert(0, COMPONENTS)

from Top_10_22_CNN_optical import CIS_Array  # noqa: E402


DEFAULT_LUT_PATH = os.path.join(HERE, 'results', 'modulcis_lut.json')


@dataclass
class ModulcisLut:
    """JSON cached power table keyed by width, height, FPS and ADC bits.

    Queries interpolate linearly over FPS at the nearest resolution and
    ADC bit depth shelf. Returns power in mW.
    """
    entries: dict

    @classmethod
    def load(cls, path: str = DEFAULT_LUT_PATH) -> 'ModulcisLut':
        with open(path) as f:
            raw = json.load(f)
        return cls(entries={
            tuple(int(x) for x in k.split(',')): float(v)
            for k, v in raw.items()
        })

    def save(self, path: str = DEFAULT_LUT_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(
                {','.join(map(str, k)): float(v) for k, v in self.entries.items()},
                f, indent=2,
            )

    def query(self, resolution: tuple[int, int], fps: float,
              adc_bits: int) -> float:
        """Look up the power in mW for a given resolution, FPS and ADC bit depth."""
        w, h = resolution
        shelf_keys = sorted({
            int(k[2]) for k in self.entries
            if int(k[0]) == w and int(k[1]) == h and int(k[3]) == adc_bits
        })
        if not shelf_keys:
            shelf_keys = sorted({
                int(k[2]) for k in self.entries
                if int(k[0]) == w and int(k[1]) == h
            })
        if not shelf_keys:
            raise KeyError(f'No LUT entry for {w}x{h}, adc={adc_bits}')

        fps_int = int(round(fps))
        lo = max([k for k in shelf_keys if k <= fps_int], default=shelf_keys[0])
        hi = min([k for k in shelf_keys if k >= fps_int], default=shelf_keys[-1])

        def _get(k):
            for key, val in self.entries.items():
                if (int(key[0]) == w and int(key[1]) == h
                        and int(key[2]) == k and int(key[3]) == adc_bits):
                    return val
            return None

        p_lo = _get(lo)
        p_hi = _get(hi)
        if p_lo is None and p_hi is None:
            raise KeyError(f'Missing LUT entry near {w}x{h} @ {fps_int} adc={adc_bits}')
        if p_lo is None:
            return p_hi
        if p_hi is None:
            return p_lo
        if hi == lo:
            return p_lo
        t = (fps_int - lo) / (hi - lo)
        return p_lo + t * (p_hi - p_lo)


def _run_cis_array(num_rows: int, num_cols: int, fps: float,
                    adc_bits: int, light_wm2: float = 5.0) -> float:
    """Build one CIS_Array instance and pull out its total power in mW."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        sensor = CIS_Array(
            feature_size_nm=65, analog_V_dd=2.8, digital_V_dd=0.8,
            input_clk_freq=20e6,
            photodiode_type=0, CFA_type=0,
            input_pixel_map=[[0.2, 1.75, 1.75]] * 5,
            pd_E=light_wm2, PD_saturation_level=0.6,
            num_rows=num_rows, num_cols=num_cols,
            Pixel_type=0, pixel_binning_map=[1, 1],
            num_PD_per_tap=1, num_of_tap=1, num_of_unused_tap=0,
            frame_rate=float(fps),
            max_subframe_rates=1, subframe_rates=1, exposure_time=0,
            IO_type=1, MUX_type=0, PGA_type=1, CDS_type=0,
            CIS_type=0, ADC_type=1,
            adc_resolution=int(adc_bits),
            PGA_DC_gain=17.38, num_mux_input=1,
            CDS_amp_gain=8, bias_voltage=0.8,
            comparator_bias_voltage=0.8,
            ADC_input_clk_freq=22e6, if_PLL=1,
            if_time_generator=1, PLL_output_frequency=44e6,
            additional_latency=0, CNN_kernel=3,
        )
    return float(sensor.system_total_power) * 1000.0  # W -> mW


def build_lut(resolutions: list[tuple[int, int]],
              fps_values: list[float],
              adc_bits_values: list[int],
              out_path: str = DEFAULT_LUT_PATH,
              light_wm2: float = 5.0,
              verbose: bool = True) -> ModulcisLut:
    """Run ModuCIS across the whole grid and save the result to JSON."""
    entries: dict[tuple, float] = {}
    total = len(resolutions) * len(fps_values) * len(adc_bits_values)
    done = 0

    for (w, h) in resolutions:
        for fps in fps_values:
            for bits in adc_bits_values:
                done += 1
                try:
                    p_mw = _run_cis_array(h, w, fps, bits, light_wm2)
                except Exception as exc:
                    if verbose:
                        print(f'  [{done:3d}/{total}] {w}x{h} @ {fps}fps '
                              f'{bits}b  FAILED: {type(exc).__name__}: {exc}')
                    continue
                entries[(w, h, int(fps), bits)] = round(p_mw, 4)
                if verbose:
                    print(f'  [{done:3d}/{total}] {w}x{h} @ {fps:6.1f}fps '
                          f'{bits:2d}b  -> {p_mw:8.2f} mW')

    lut = ModulcisLut(entries=entries)
    lut.save(out_path)
    if verbose:
        print(f'\n[ModuCIS LUT] saved {len(entries)} entries to {out_path}')
    return lut


if __name__ == '__main__':
    resolutions = [(640, 480), (1920, 1080), (1920, 1200)]
    fps_values  = [5, 15, 30, 60, 90, 120]
    adc_bits_values = [8, 10, 12]
    print('[ModuCIS LUT] building...')
    build_lut(resolutions, fps_values, adc_bits_values)
