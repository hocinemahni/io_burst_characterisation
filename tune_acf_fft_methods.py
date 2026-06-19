#!/usr/bin/env python3
"""Generate ACF/FFT candidates for comparison.

Run:
  python tune_acf_fft_methods.py --input-dir logs --output-dir acf_fft_candidates --clean

It tries several ACF/FFT reconstruction choices and writes:
  - candidates_summary.csv
  - candidate ACF/FFT figures for NAMD, E3SM, HACC
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import reproduce_article_figures as R


def acf_values(y, max_lag=50):
    return R.autocorrelation(pd.Series(y), max_lag=max_lag)


def fft_values(y, dt=1.0):
    y = pd.to_numeric(pd.Series(y), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(y) < 3:
        return pd.DataFrame({"frequency_hz": [], "power": []}), {}
    y = (y - y.mean()) / (y.std(ddof=1) + 1e-12)
    vals = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), d=dt)
    mask = freqs > 0
    freqs = freqs[mask]
    power = np.abs(vals[mask]) ** 2
    sp = pd.DataFrame({"frequency_hz": freqs, "power": power})
    if sp.empty:
        return sp, {}
    i = int(sp["power"].idxmax())
    return sp, {"dominant_frequency_hz": float(sp.loc[i, "frequency_hz"]), "max_power": float(sp.loc[i, "power"])}


def build_candidates(app, posix, dxt, default_events):
    out = {}
    if default_events is not None and not default_events.empty:
        out["default_events"] = default_events
        out["raw_sequence_default"] = R._events_to_raw_sequence(default_events)

    if posix is not None and not posix.empty:
        out["posix_article_records"] = R.posix_to_article_events(posix)
        out["posix_timebin_1s"] = R.posix_to_time_binned_events(posix, bin_width=1.0, nodes=1.0)
        out["posix_timebin_0p5s"] = R.posix_to_time_binned_events(posix, bin_width=0.5, nodes=1.0)

    if dxt is not None and not dxt.empty:
        de = R.dxt_to_events(dxt)
        out["dxt_raw_sequence"] = R._events_to_raw_sequence(de)
        for bw in [0.000625, 0.001, 0.005, 0.01, 0.1, 1.0, 2.0, 5.0]:
            out[f"dxt_volume_bin_{bw:g}s"] = R._events_to_regular_volume_bins(de, bw)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("logs"))
    ap.add_argument("--output-dir", type=Path, default=Path("acf_fft_candidates"))
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figdir = args.output_dir / "figures"
    figdir.mkdir(exist_ok=True)

    rows = []
    for path in sorted(args.input_dir.glob("*.darshan")):
        posix, dxt, meta = R.read_darshan_log(path)
        app = str(meta["application"])
        default = R.choose_events(app, R.posix_to_article_events(posix), R.dxt_to_events(dxt), use_dxt_for=["NAMD", "HACC", "IOR"])
        candidates = build_candidates(app, posix, dxt, default)

        for name, series in candidates.items():
            if series is None or series.empty or "bandwidth_mb_s" not in series:
                continue
            source = series.get("source", pd.Series([name])).iloc[0] if len(series) else name
            acf = acf_values(series["bandwidth_mb_s"], 50)
            dt = R.infer_sampling_period_for_series(series) if hasattr(R, "infer_sampling_period_for_series") else 1.0
            sp, ft = fft_values(series["bandwidth_mb_s"], dt=dt)

            rows.append({
                "application": app,
                "candidate": name,
                "n": len(series),
                "source": source,
                "acf_lag1": float(acf["acf"].iloc[1]) if len(acf) > 1 else np.nan,
                "acf_abs_mean_lag1_50": float(acf["acf"].iloc[1:].abs().mean()) if len(acf) > 1 else np.nan,
                "dominant_frequency_hz": ft.get("dominant_frequency_hz", np.nan),
                "max_power": ft.get("max_power", np.nan),
            })

    pd.DataFrame(rows).to_csv(args.output_dir / "candidates_summary.csv", index=False)
    print("Wrote", args.output_dir / "candidates_summary.csv")


if __name__ == "__main__":
    main()
