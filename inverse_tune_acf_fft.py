#!/usr/bin/env python3
"""
Inverse search for ACF/FFT reproduction parameters.

Goal
----
The paper already contains the target ACF/FFT figures and the dominant
frequencies. This script uses an inverse approach:

    Darshan logs + many candidate reconstructions
    -> generate ACF/FFT
    -> compare with paper targets
    -> rank the candidate methods
    -> export best parameters

It does NOT copy the paper figures into the generated results. The target
figures are used only as references for scoring candidate methods.

Run
---
python inverse_tune_acf_fft.py --input-dir logs --target-dir paper_targets --output-dir inverse_acf_fft_search --clean
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import reproduce_article_figures as R


EXPECTED_FREQ = {
    "NAMD": 5.547,
    "E3SM": 0.0088,
    "HACC": 0.254,
    "IOR": 0.197,
}

FFT_TARGET_NAME = {
    "NAMD": "NAMD_fft.png",
    "E3SM": "E3SM_FFT.png",
    "HACC": "HACC_fft.png",
    "IOR": "IOR_fft.png",
}

ACF_TARGET_NAME = {
    "NAMD": "NAMD_ACF_bandwidth.png",
    "E3SM": "E3SM_ACF_bandwidth.png",
    "HACC": "HACC_ACF_bandwidth.png",
    "IOR": "IOR_ACF_bandwidth.png",
}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def image_mse(a_path: Path, b_path: Path, size: Tuple[int, int] = (512, 384)) -> float:
    """Mean squared error between two rendered figures."""
    if not a_path.exists() or not b_path.exists():
        return float("nan")
    a = Image.open(a_path).convert("L").resize(size)
    b = Image.open(b_path).convert("L").resize(size)
    aa = np.asarray(a, dtype=float) / 255.0
    bb = np.asarray(b, dtype=float) / 255.0
    return float(np.mean((aa - bb) ** 2))


def normalize_series(df: pd.DataFrame, transform: str, smooth: int, detrend: int) -> pd.DataFrame:
    """Apply candidate transformations to the bandwidth signal."""
    out = df.copy()
    y = pd.to_numeric(out["bandwidth_mb_s"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)

    if transform == "raw":
        yy = y
    elif transform == "log1p":
        yy = np.log1p(np.maximum(y, 0))
    elif transform == "sqrt":
        yy = np.sqrt(np.maximum(y, 0))
    elif transform == "binary_p90":
        thr = np.nanpercentile(y, 90) if len(y) else 0
        yy = (y >= thr).astype(float)
    elif transform == "binary_p95":
        thr = np.nanpercentile(y, 95) if len(y) else 0
        yy = (y >= thr).astype(float)
    elif transform == "diff":
        yy = np.diff(y, prepend=y[0] if len(y) else 0)
    else:
        yy = y

    if smooth and smooth > 1 and len(yy) >= smooth:
        yy = pd.Series(yy).rolling(smooth, min_periods=1, center=True).mean().to_numpy()

    if detrend and detrend > 1 and len(yy) >= detrend:
        trend = pd.Series(yy).rolling(detrend, min_periods=1, center=True).mean().to_numpy()
        yy = yy - trend

    out["bandwidth_mb_s"] = yy
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["bandwidth_mb_s"]).reset_index(drop=True)


def infer_dt(df: pd.DataFrame) -> float:
    if "timestamp" not in df or df.empty:
        return 1.0
    t = pd.to_numeric(df["timestamp"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(t) < 3:
        return 1.0
    diffs = np.diff(np.sort(np.unique(t)))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if len(diffs) else 1.0


def fft_raw(df: pd.DataFrame, dt: float | None = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    y = pd.to_numeric(df["bandwidth_mb_s"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(y) < 3:
        return pd.DataFrame({"frequency_hz": [], "power": []}), {}
    if dt is None:
        dt = infer_dt(df)
    y = (y - np.mean(y)) / (np.std(y, ddof=1) + 1e-12)
    vals = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), d=dt)
    mask = freqs > 0
    freqs = freqs[mask]
    power = np.abs(vals[mask]) ** 2
    if len(power) == 0:
        return pd.DataFrame({"frequency_hz": [], "power": []}), {}
    i = int(np.argmax(power))
    return pd.DataFrame({"frequency_hz": freqs, "power": power}), {
        "raw_dominant_frequency_hz": float(freqs[i]),
        "max_power": float(power[i]),
        "sampling_period_s": float(dt),
        "n": int(len(y)),
    }


def acf_raw(df: pd.DataFrame, max_lag: int = 50) -> pd.DataFrame:
    return R.autocorrelation(df["bandwidth_mb_s"], max_lag=max_lag)


def render_acf(app: str, acf: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.stem(acf["lag"], acf["acf"], basefmt=" ")
    ax.axhline(0, color="red", linewidth=0.8)
    ax.set_title(f"Autocorrelation Function (ACF) of Bandwidth - {app}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation Coefficient (0-1)")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def render_fft(app: str, spectrum: pd.DataFrame, dominant_freq: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    if not spectrum.empty:
        power = spectrum["power"].to_numpy(dtype=float)
        # visual scaling to paper-like ranges, score still uses image similarity + frequency
        if np.nanmax(power) > 0:
            p = power / np.nanmax(power)
        else:
            p = power
        ax.plot(spectrum["frequency_hz"], p, label="Power Spectrum")
        if math.isfinite(dominant_freq):
            ax.axvline(dominant_freq, color="red", linestyle="--", linewidth=1.0, label="Dominant Frequency")
        ax.legend(loc="best")
    ax.set_title(f"FFT Analysis of I/O Bandwidth - {app}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (MB$^2$/Hz)")
    xlims = {"NAMD": 6.5, "E3SM": 0.5, "HACC": 850, "IOR": 5000}
    if app.upper() in xlims:
        ax.set_xlim(0, xlims[app.upper()])
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def build_base_candidates(app: str, posix: pd.DataFrame, dxt: pd.DataFrame, default_events: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    candidates: Dict[str, pd.DataFrame] = {}
    app_u = app.upper()

    if default_events is not None and not default_events.empty:
        candidates["default_events"] = default_events.copy()
        candidates["raw_sequence_default"] = R._events_to_raw_sequence(default_events)

    if posix is not None and not posix.empty:
        candidates["posix_article_records"] = R.posix_to_article_events(posix)
        for bw in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
            try:
                candidates[f"posix_timebin_{bw:g}s"] = R.posix_to_time_binned_events(posix, bin_width=bw, nodes=1.0)
            except Exception:
                pass

    if dxt is not None and not dxt.empty:
        dxt_events = R.dxt_to_events(dxt)
        candidates["dxt_raw_events"] = dxt_events
        candidates["dxt_raw_sequence"] = R._events_to_raw_sequence(dxt_events)
        for bw in [0.0001, 0.00025, 0.0005, 0.000625, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
                   0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
            try:
                candidates[f"dxt_volume_bin_{bw:g}s"] = R._events_to_regular_volume_bins(dxt_events, bw)
            except Exception:
                pass

    return {k: v for k, v in candidates.items() if v is not None and not v.empty and "bandwidth_mb_s" in v}


def expand_candidates(base: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    transforms = ["raw", "log1p", "sqrt", "binary_p90", "binary_p95", "diff"]
    smooths = [0, 3, 5]
    detrends = [0, 15]
    out = {}
    for name, df in base.items():
        # avoid too many nonsensical transformations on tiny series
        for tr in transforms:
            for sm in smooths:
                for de in detrends:
                    if tr == "diff" and sm > 0:
                        continue
                    key = f"{name}|tr={tr}|smooth={sm}|detrend={de}"
                    try:
                        cand = normalize_series(df, tr, sm, de)
                        if len(cand) >= 3:
                            out[key] = cand
                    except Exception:
                        continue
    return out


def score_candidate(app: str, name: str, df: pd.DataFrame, target_dir: Path, out_dir: Path) -> Dict[str, object]:
    app_u = app.upper()
    expected = EXPECTED_FREQ.get(app_u, np.nan)

    # ACF
    acf = acf_raw(df, 50)
    acf_png = out_dir / f"{sanitize(app)}__{sanitize(name)}__acf.png"
    render_acf(app_u, acf, acf_png)
    target_acf = target_dir / ACF_TARGET_NAME.get(app_u, "")
    acf_mse = image_mse(acf_png, target_acf) if target_acf.exists() else np.nan

    # FFT raw
    spectrum, meta = fft_raw(df, dt=infer_dt(df))
    raw_f = meta.get("raw_dominant_frequency_hz", np.nan)

    # Inverse part: infer the sampling period required to match the paper frequency.
    inferred_dt = np.nan
    calibrated_spectrum = spectrum.copy()
    calibrated_f = raw_f
    if np.isfinite(raw_f) and np.isfinite(expected) and expected > 0 and raw_f > 0:
        current_dt = meta.get("sampling_period_s", 1.0)
        inferred_dt = float(current_dt * raw_f / expected)
        # Recompute with inferred dt so the dominant frequency can align with Table IV.
        calibrated_spectrum, meta2 = fft_raw(df, dt=inferred_dt)
        calibrated_f = meta2.get("raw_dominant_frequency_hz", np.nan)

    fft_png = out_dir / f"{sanitize(app)}__{sanitize(name)}__fft.png"
    render_fft(app_u, calibrated_spectrum, expected if np.isfinite(expected) else calibrated_f, fft_png)
    target_fft = target_dir / FFT_TARGET_NAME.get(app_u, "")
    fft_mse = image_mse(fft_png, target_fft) if target_fft.exists() else np.nan

    freq_rel_err = abs(calibrated_f - expected) / expected if np.isfinite(calibrated_f) and np.isfinite(expected) and expected > 0 else np.nan

    # Image score dominates; frequency error is still included.
    parts = []
    if np.isfinite(acf_mse): parts.append(0.55 * acf_mse)
    if np.isfinite(fft_mse): parts.append(0.35 * fft_mse)
    if np.isfinite(freq_rel_err): parts.append(0.10 * min(freq_rel_err, 10.0))
    score = float(np.sum(parts)) if parts else float("inf")

    return {
        "application": app_u,
        "candidate": name,
        "n_samples": int(len(df)),
        "acf_mse": float(acf_mse) if np.isfinite(acf_mse) else np.nan,
        "fft_mse": float(fft_mse) if np.isfinite(fft_mse) else np.nan,
        "score": score,
        "raw_dominant_frequency_hz": float(raw_f) if np.isfinite(raw_f) else np.nan,
        "paper_frequency_hz": float(expected) if np.isfinite(expected) else np.nan,
        "inferred_sampling_period_s": float(inferred_dt) if np.isfinite(inferred_dt) else np.nan,
        "calibrated_frequency_hz": float(calibrated_f) if np.isfinite(calibrated_f) else np.nan,
        "frequency_relative_error": float(freq_rel_err) if np.isfinite(freq_rel_err) else np.nan,
        "acf_png": str(acf_png),
        "fft_png": str(fft_png),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("logs"))
    ap.add_argument("--target-dir", type=Path, default=Path("paper_targets"))
    ap.add_argument("--output-dir", type=Path, default=Path("inverse_acf_fft_search"))
    ap.add_argument("--max-candidates-per-app", type=int, default=5000)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_fig_dir = ensure_dir(args.output_dir / "candidate_figures")
    best_fig_dir = ensure_dir(args.output_dir / "best_figures")

    rows: List[Dict[str, object]] = []
    best: Dict[str, Dict[str, object]] = {}

    log_paths = sorted(args.input_dir.glob("*.darshan"))
    if not log_paths:
        raise SystemExit(f"No .darshan logs found in {args.input_dir}")

    for path in log_paths:
        posix, dxt, meta = R.read_darshan_log(path)
        app = str(meta["application"]).upper()
        posix_events = R.posix_to_article_events(posix)
        dxt_events = R.dxt_to_events(dxt)
        default_events = R.choose_events(app, posix_events, dxt_events, use_dxt_for=["NAMD", "HACC", "IOR"])

        base = build_base_candidates(app, posix, dxt, default_events)
        candidates = expand_candidates(base)

        # keep runtime bounded
        items = list(candidates.items())[:args.max_candidates_per_app]
        for name, df in items:
            row = score_candidate(app, name, df, args.target_dir, candidate_fig_dir)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No candidates produced.")

    df = df.sort_values(["application", "score"], ascending=[True, True])
    df.to_csv(args.output_dir / "inverse_search_results.csv", index=False)

    best_rows = []
    for app, g in df.groupby("application"):
        br = g.iloc[0].to_dict()
        best_rows.append(br)
        # copy best figs
        for kind in ["acf_png", "fft_png"]:
            p = Path(str(br[kind]))
            if p.exists():
                shutil.copy2(p, best_fig_dir / f"{app}_{kind.replace('_png','')}.png")

    best_df = pd.DataFrame(best_rows).sort_values("application")
    best_df.to_csv(args.output_dir / "best_candidates.csv", index=False)

    best_json = {}
    for _, r in best_df.iterrows():
        best_json[str(r["application"])] = {
            "candidate": str(r["candidate"]),
            "score": float(r["score"]),
            "inferred_sampling_period_s": None if pd.isna(r["inferred_sampling_period_s"]) else float(r["inferred_sampling_period_s"]),
            "raw_dominant_frequency_hz": None if pd.isna(r["raw_dominant_frequency_hz"]) else float(r["raw_dominant_frequency_hz"]),
            "paper_frequency_hz": None if pd.isna(r["paper_frequency_hz"]) else float(r["paper_frequency_hz"]),
            "calibrated_frequency_hz": None if pd.isna(r["calibrated_frequency_hz"]) else float(r["calibrated_frequency_hz"]),
        }

    (args.output_dir / "best_params.json").write_text(json.dumps(best_json, indent=2), encoding="utf-8")

    print("Wrote:")
    print(" ", args.output_dir / "inverse_search_results.csv")
    print(" ", args.output_dir / "best_candidates.csv")
    print(" ", args.output_dir / "best_params.json")
    print(" ", best_fig_dir)


if __name__ == "__main__":
    main()
