#!/usr/bin/env python3
"""
Reproduce the scientific analysis and figures for non-Darshan CSV workloads.
Supports:
  - Incompact3D (logs/incompact3d/incompact3d_io.csv)
  - LAMMPS (logs/lammps/lammps_io.csv)
  - LifeScience (logs/lifescience/lifescience_request.csv)
  - QuantumEspresso-CP (logs/quantumespresso-cp/quantumespresso-cp_request.csv)
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Force non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MB = 1024.0 * 1024.0

CSV_APPS = {
    "Incompact3D": "incompact3d/incompact3d_io.csv",
    "LAMMPS": "lammps/lammps_io.csv",
    "LifeScience": "lifescience/lifescience_request.csv",
    "QuantumEspresso-CP": "quantumespresso-cp/quantumespresso-cp_request.csv"
}

# ---------------------------------------------------------------------------
# Statistics and Core Mathematics (copied from Darshan pipeline)
# ---------------------------------------------------------------------------
def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan
    if method == "spearman":
        data = data.rank(method="average")
    return float(data["x"].corr(data["y"], method="pearson"))


def shannon_entropy(values: pd.Series, bins: int = 10) -> Tuple[float, float]:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(arr) == 0:
        return np.nan, np.nan
    counts, _ = np.histogram(arr, bins=bins)
    p = counts[counts > 0].astype(float)
    p /= p.sum()
    h = float(-(p * np.log2(p)).sum())
    h_rel = h / np.log2(bins) if bins > 1 else np.nan
    return h, float(h_rel)


def autocorrelation(values: pd.Series, max_lag: int = 50) -> pd.DataFrame:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return pd.DataFrame({"lag": [], "acf": []})
    x = x - x.mean()
    denom = np.dot(x, x)
    max_lag = min(max_lag, len(x) - 1)
    rows = []
    for lag in range(max_lag + 1):
        acf = np.nan if denom == 0 else float(np.dot(x[:len(x)-lag], x[lag:]) / denom)
        rows.append({"lag": lag, "acf": acf})
    return pd.DataFrame(rows)


def fft_spectrum(events: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = events.sort_values("timestamp")
    y = pd.to_numeric(df["bandwidth_mb_s"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    if len(y) < 3:
        return pd.DataFrame({"frequency_hz": [], "power": []}), {
            "top_dominant_frequency_hz": np.nan,
            "detected_period_s": np.nan,
            "cyclic_strength": np.nan,
        }
    t = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).to_numpy()
    uniq = np.unique(t)
    dt = np.median(np.diff(uniq)) if len(uniq) > 2 else 1.0
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0
    y_norm = (y - y.mean()) / (y.std(ddof=1) + 1e-12)
    vals = np.fft.fft(y_norm)
    freqs = np.fft.fftfreq(len(y_norm), d=dt)
    mask = freqs > 0
    freqs = freqs[mask]
    power = np.abs(vals[mask]) ** 2
    if len(power) == 0:
        return pd.DataFrame({"frequency_hz": [], "power": []}), {
            "top_dominant_frequency_hz": np.nan,
            "detected_period_s": np.nan,
            "cyclic_strength": np.nan,
        }
    idx = int(np.argmax(power))
    fdom = float(freqs[idx])
    spectrum = pd.DataFrame({"frequency_hz": freqs, "power": power})
    features = {
        "top_dominant_frequency_hz": fdom,
        "detected_period_s": float(1.0 / fdom) if fdom > 0 else np.nan,
        "cyclic_strength": float(np.sqrt(power.max()) / (np.sqrt(power).mean() + 1e-12)),
    }
    return spectrum, features


def detect_adaptive(
    df: pd.DataFrame,
    window: int = 15,
    fixed_percentile: float = 95.0,
    adaptive_rule: str = "bw_only",
    k_low: float = 0.8,
    k_high: float = 1.2,
    k_extra: float = 0.3
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = df.sort_values("timestamp").reset_index(drop=True).copy()
    if out.empty:
        return out, {}
    if len(out) < max(3, window):
        window = max(3, min(window, len(out)))

    h_bw, rel_bw = shannon_entropy(out["bandwidth_mb_s"])
    h_io, rel_io = shannon_entropy(out["io_size_mb"])
    _, fft_features = fft_spectrum(out)
    mean_rel = np.nanmean([rel_bw, rel_io])
    cyclic_strength = fft_features.get("cyclic_strength", np.nan)

    if mean_rel > 0.5 and cyclic_strength > 3:
        k = k_high + k_extra
    elif mean_rel > 0.5 or cyclic_strength > 3:
        k = k_high
    else:
        k = k_low

    min_periods = min(max(3, window), len(out))
    out["mu_bw"] = out["bandwidth_mb_s"].rolling(window, min_periods=min_periods).mean()
    out["sigma_bw"] = out["bandwidth_mb_s"].rolling(window, min_periods=min_periods).std(ddof=1)
    out["mu_io"] = out["io_size_mb"].rolling(window, min_periods=min_periods).mean()
    out["sigma_io"] = out["io_size_mb"].rolling(window, min_periods=min_periods).std(ddof=1)
    out["k_dynamic"] = k
    out["threshold_bw_adaptive"] = out["mu_bw"] + k * out["sigma_bw"]
    out["threshold_io_adaptive"] = out["mu_io"] + k * out["sigma_io"]

    out["burst_adaptive_dual"] = (
        (out["bandwidth_mb_s"] > out["threshold_bw_adaptive"]) &
        (out["io_size_mb"] > out["threshold_io_adaptive"])
    )
    out["burst_adaptive_bw_only"] = out["bandwidth_mb_s"] > out["threshold_bw_adaptive"]
    
    if adaptive_rule == "dual":
        out["burst_adaptive"] = out["burst_adaptive_dual"]
    else:
        out["burst_adaptive"] = out["burst_adaptive_bw_only"]
        
    fixed = float(out["bandwidth_mb_s"].quantile(fixed_percentile / 100.0))
    out["threshold_bw_fixed"] = fixed
    out["burst_fixed"] = out["bandwidth_mb_s"] > fixed

    intersection = int((out["burst_adaptive"] & out["burst_fixed"]).sum())
    union = int((out["burst_adaptive"] | out["burst_fixed"]).sum())
    meta = {
        "k_dynamic": k,
        "fixed_percentile": fixed_percentile,
        "fixed_threshold_bw": fixed,
        "adaptive_burst_count": int(out["burst_adaptive"].sum()),
        "fixed_burst_count": int(out["burst_fixed"].sum()),
        "iou_adaptive_vs_fixed": float(intersection / union) if union else 0.0,
        "relative_unpredictability_bw_percent": float(100 * rel_bw),
        "relative_unpredictability_io_size_percent": float(100 * rel_io),
        "cyclic_strength": float(cyclic_strength),
    }
    return out, meta


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------
def setup_axes(ax, title: str, xlabel: str, ylabel: str):
    title = str(title)
    if len(title) > 42 and " - " in title:
        left, right = title.rsplit(" - ", 1)
        title = left + "\n- " + right
    ax.set_title(title, fontsize=13, pad=8, wrap=True)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.28)


def style_acf_stem(ax, markerline, stemlines, baseline=None):
    try:
        plt.setp(markerline, markersize=3.6, marker="o", markeredgewidth=0.6)
    except Exception:
        pass
    try:
        plt.setp(stemlines, linewidth=0.8)
    except Exception:
        pass
    if baseline is not None:
        try:
            plt.setp(baseline, visible=False)
        except Exception:
            pass


def plot_detection_on_ax(ax, app: str, detected: pd.DataFrame):
    df = detected.sort_values("timestamp").reset_index(drop=True).copy()
    t = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0)
    bw = pd.to_numeric(df["bandwidth_mb_s"], errors="coerce").fillna(0.0)
    T_fixed = float(pd.to_numeric(df["threshold_bw_fixed"], errors="coerce").dropna().iloc[0])

    # 1. Bandwidth line
    ax.plot(t, bw, color="#5dade2", alpha=0.40, linewidth=0.8, label="Bandwidth", zorder=1)

    # 2. Fixed threshold line
    ax.axhline(T_fixed, color="steelblue", linestyle="-", linewidth=1.1, label=f"Fixed Threshold = {T_fixed:.1f} MB/s", zorder=3)

    # 3. Burst markers
    mask_adapt = df["burst_adaptive"].astype(bool) if "burst_adaptive" in df.columns else pd.Series(False, index=df.index)
    mask_fixed = df["burst_fixed"].astype(bool) if "burst_fixed" in df.columns else (bw > T_fixed)

    # Crimson dots for adaptive bursts
    if mask_adapt.any():
        ax.scatter(t[mask_adapt], bw[mask_adapt], color="crimson", s=14, marker="o", edgecolors="none", alpha=0.80, label="Adaptive Bursts", zorder=4)

    # Steelblue crosses for fixed bursts
    if mask_fixed.any():
        ax.scatter(t[mask_fixed], bw[mask_fixed], color="steelblue", marker="x", s=18, linewidths=1.0, alpha=0.90, label="Fixed Bursts", zorder=5)

    setup_axes(ax, f"Detected I/O Bursts - {app}", "Timestamp (s)", "Bandwidth (MB/s)")
    ax.legend(fontsize=11, loc="best", framealpha=0.9)


def combined_grid(apps: List[str], plot_func, title: str, outpath: Path, ncols: int = 2, figsize=None):
    if not apps:
        return
    n = len(apps)
    nrows = math.ceil(n / ncols)
    if figsize is None:
        figsize = (7.5 * ncols, 3.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.asarray(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis("off")
    for ax, app in zip(axes, apps):
        plot_func(ax, app)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV Log Loader & Parser
# ---------------------------------------------------------------------------
def load_and_bin_csv(path: Path, app_name: str) -> pd.DataFrame:
    print(f"Loading and processing {path.name} for {app_name}...")
    df = pd.read_csv(path)
    
    # Strip whitespace from column names and string values
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Match case-insensitive column headers
    ts_col, type_col, file_col, start_col, end_col = None, None, None, None, None
    for col in df.columns:
        lcol = col.lower().replace("_", "")
        if lcol in ["timestamp", "time", "ts"]:
            ts_col = col
        elif lcol in ["requesttype", "type", "op", "operation", "request_type"]:
            type_col = col
        elif lcol in ["filename", "file", "path"]:
            file_col = col
        elif lcol in ["offsetstart", "start", "offset_start", "offset"]:
            start_col = col
        elif lcol in ["offsetend", "end", "offset_end", "length", "size"]:
            end_col = col

    if not ts_col:
        raise ValueError(f"Could not find timestamp column in {path}")

    # Standardize DataFrame columns
    norm_df = pd.DataFrame()
    norm_df["timestamp_raw"] = pd.to_numeric(df[ts_col], errors="coerce").fillna(0.0)
    norm_df["io_type_raw"] = df[type_col].astype(str).str.lower() if type_col else "write"
    norm_df["filename"] = df[file_col].astype(str) if file_col else "unknown_file"
    
    start_vals = pd.to_numeric(df[start_col], errors="coerce").fillna(0.0) if start_col else 0.0
    end_vals = pd.to_numeric(df[end_col], errors="coerce").fillna(0.0) if end_col else 0.0

    # Calculate operation sizes
    if end_col and ("length" in end_col.lower() or "size" in end_col.lower()):
        norm_df["size_bytes"] = end_vals.abs()
    else:
        norm_df["size_bytes"] = (end_vals - start_vals).abs()

    # Detect unit (ms vs s)
    t_max = norm_df["timestamp_raw"].max()
    if t_max > 1e11:
        norm_df["timestamp"] = norm_df["timestamp_raw"] / 1000.0
    else:
        norm_df["timestamp"] = norm_df["timestamp_raw"]

    # Normalize time start to 0.0
    t_min = norm_df["timestamp"].min()
    norm_df["timestamp"] = norm_df["timestamp"] - t_min

    duration = norm_df["timestamp"].max()
    if duration <= 0:
        duration = 1.0

    # Compute adaptive bin width targeting ~700 bins
    raw_bin_width = duration / 700.0
    if raw_bin_width < 0.1:
        bin_width = 0.05
    elif raw_bin_width < 1.0:
        bin_width = round(raw_bin_width, 2)
    elif raw_bin_width < 10.0:
        bin_width = round(raw_bin_width, 1)
    else:
        bin_width = round(raw_bin_width, 0)
    bin_width = max(bin_width, 0.01)

    print(f"  Trace duration: {duration:.2f} s. Selected bin width: {bin_width:.2f} s.")

    # Aggregate into temporal bins
    n_bins = int(math.ceil(duration / bin_width)) + 1
    bin_starts = np.arange(n_bins, dtype=float) * bin_width

    bytes_total = np.zeros(n_bins, dtype=float)
    bytes_write = np.zeros(n_bins, dtype=float)
    bytes_read = np.zeros(n_bins, dtype=float)
    ops_total = np.zeros(n_bins, dtype=float)
    unique_files_per_bin = [set() for _ in range(n_bins)]

    for _, r in norm_df.iterrows():
        t_val = r["timestamp"]
        b_idx = int(np.floor(t_val / bin_width))
        if 0 <= b_idx < n_bins:
            size = r["size_bytes"]
            bytes_total[b_idx] += size
            ops_total[b_idx] += 1
            if r["io_type_raw"] in ["write", "w"]:
                bytes_write[b_idx] += size
            else:
                bytes_read[b_idx] += size
            unique_files_per_bin[b_idx].add(r["filename"])

    concurrent_io = np.array([len(s) if len(s) > 0 else 1.0 for s in unique_files_per_bin])
    bw_mb_s = (bytes_total / MB) / bin_width
    io_size_mb = np.divide(bytes_total / MB, ops_total, out=np.zeros_like(bytes_total), where=ops_total > 0)
    io_type = np.where(bytes_total <= 0, "idle", np.where(bytes_write >= bytes_read, "write", "read"))

    binned_df = pd.DataFrame({
        "application": app_name,
        "source": "CSV_RECONSTRUCTED",
        "record_id": np.arange(n_bins, dtype=np.int64),
        "rank": -1,
        "io_type": io_type,
        "timestamp": bin_starts,
        "end_time": bin_starts + bin_width,
        "duration": bin_width,
        "io_size_mb": io_size_mb,
        "total_size_mb": bytes_total / MB,
        "operation_count": ops_total,
        "offset": 0.0,
        "concurrent_io": concurrent_io,
        "bandwidth_mb_s": bw_mb_s
    })
    return binned_df


# ---------------------------------------------------------------------------
# Numerical Table Calculation
# ---------------------------------------------------------------------------
def compute_tables_csv(events_by_app: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows, corr_rows, entropy_rows, fft_rows = [], [], [], []
    apps_sorted = sorted(events_by_app.keys())
    for app in apps_sorted:
        df = events_by_app[app]
        if df.empty:
            continue
        bw = df["bandwidth_mb_s"]
        io = df["io_size_mb"]
        
        summary_rows.append({
            "application": app,
            "binned_records": int(len(df)),
            "mean_bandwidth_mb_s": float(bw.mean()),
            "std_bandwidth_mb_s": float(bw.std(ddof=1)),
            "bandwidth_std_over_mean_percent": float(bw.std(ddof=1) / bw.mean() * 100) if bw.mean() else np.nan,
            "mean_io_size_mb": float(io.mean()),
            "std_io_size_mb": float(io.std(ddof=1)),
            "io_size_std_over_mean_percent": float(io.std(ddof=1) / io.mean() * 100) if io.mean() else np.nan,
        })
        
        io_type_num = df["io_type"].map({"idle": 0, "read": 1, "write": 2}).astype(float)
        corr_rows.append({
            "application": app,
            "pearson_bw_io_size": safe_corr(io, bw, "pearson"),
            "spearman_bw_io_size": safe_corr(io, bw, "spearman"),
            "pearson_bw_io_type": safe_corr(io_type_num, bw, "pearson"),
            "spearman_bw_io_type": safe_corr(io_type_num, bw, "spearman"),
            "pearson_bw_concurrent_io": safe_corr(df["concurrent_io"], bw, "pearson"),
            "spearman_bw_concurrent_io": safe_corr(df["concurrent_io"], bw, "spearman"),
        })
        
        h_bw, rel_bw = shannon_entropy(bw)
        h_io, rel_io = shannon_entropy(io)
        entropy_rows.append({
            "application": app,
            "shannon_entropy_bw": h_bw,
            "relative_unpredictability_bw_percent": 100 * rel_bw,
            "shannon_entropy_io_size": h_io,
            "relative_unpredictability_io_size_percent": 100 * rel_io,
        })
        
        _, fft_features = fft_spectrum(df)
        fft_features["application"] = app
        fft_rows.append(fft_features)

    return pd.DataFrame(summary_rows), pd.DataFrame(corr_rows), pd.DataFrame(entropy_rows), pd.DataFrame(fft_rows)


# ---------------------------------------------------------------------------
# Main reproduction execution
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Reproduce burst prediction metrics and plots for non-Darshan CSV logs.")
    parser.add_argument("--input-dir", type=Path, default=Path("logs"), help="Logs base directory containing CSV subdirs.")
    parser.add_argument("--output-dir", type=Path, default=Path("csv_results"), help="Directory where outputs will be saved.")
    parser.add_argument("--window", type=int, default=15, help="Sliding window size for adaptive prediction model.")
    parser.add_argument("--fixed-percentile", type=float, default=95.0, help="Fixed threshold percentile baseline.")
    parser.add_argument("--clean", action="store_true", help="Clean output directory before running.")
    args = parser.parse_args()

    if args.clean and args.output_dir.exists():
        print(f"Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    # Setup directories
    figdir = args.output_dir / "figures"
    csvdir = args.output_dir / "csv"
    for p in [figdir, csvdir, figdir / "acf", figdir / "fft", figdir / "abpm"]:
        p.mkdir(parents=True, exist_ok=True)

    events_by_app: Dict[str, pd.DataFrame] = {}
    detections_by_app: Dict[str, pd.DataFrame] = {}
    detection_meta_rows = []

    # Process each application CSV file
    for app, rel_path in CSV_APPS.items():
        full_path = args.input_dir / rel_path
        if not full_path.exists():
            print(f"WARNING: CSV trace for workload '{app}' not found at: {full_path}. Skipping.")
            continue
        
        try:
            binned_df = load_and_bin_csv(full_path, app)
            events_by_app[app] = binned_df
            binned_df.to_csv(csvdir / f"{app}_analysis_events.csv", index=False)

            # Individual figures
            # Bandwidth variation
            fig, ax = plt.subplots(figsize=(7.0, 3.8))
            ax.plot(binned_df["timestamp"], binned_df["bandwidth_mb_s"], color="tab:blue", linewidth=0.8, alpha=0.65)
            setup_axes(ax, f"Bandwidth Variation - {app}", "Time (s)", "Bandwidth (MB/s)")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(figdir / f"{app}_bandwidth_variation.png", dpi=300)
            plt.close(fig)

            # CDF & PDF
            x = binned_df["bandwidth_mb_s"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if len(x) > 0:
                xs = np.sort(x)
                cdf = np.arange(1, len(xs) + 1) / len(xs)
                fig, ax = plt.subplots(figsize=(4.8, 3.4))
                ax.plot(xs, cdf, color="tab:blue", linewidth=1.3)
                setup_axes(ax, f"Bandwidth CDF of {app}", "Bandwidth (MB/s)", "CDF")
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                fig.savefig(figdir / f"{app}_CDF_bandwidth.png", dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(4.8, 3.4))
                ax.hist(x, bins=30, color="tab:blue", alpha=0.9)
                setup_axes(ax, f"Bandwidth PDF of {app}", "Bandwidth (MB/s)", "Frequency")
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                fig.savefig(figdir / f"{app}_PDF_bandwidth.png", dpi=300)
                plt.close(fig)

            # ACF
            acf = autocorrelation(binned_df["bandwidth_mb_s"], max_lag=50)
            fig, ax = plt.subplots(figsize=(4.8, 3.4))
            markerline, stemlines, baseline = ax.stem(acf["lag"], acf["acf"], basefmt=" ")
            style_acf_stem(ax, markerline, stemlines, baseline)
            ax.axhline(0, color="black", linewidth=0.8)
            setup_axes(ax, f"ACF Bandwidth of {app}", "Lag", "ACF")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(figdir / "acf" / f"{app}_ACF_bandwidth.png", dpi=300)
            plt.close(fig)

            # FFT
            spectrum, features = fft_spectrum(binned_df)
            fig, ax = plt.subplots(figsize=(4.8, 3.4))
            if not spectrum.empty:
                ax.plot(spectrum["frequency_hz"], spectrum["power"], color="tab:blue", linewidth=1.0, label="Power Spectrum")
                if math.isfinite(features["top_dominant_frequency_hz"]):
                    ax.axvline(features["top_dominant_frequency_hz"], color="red", linestyle="--", linewidth=1.0, label="Dominant Frequency")
                    ax.legend(loc="best")
            setup_axes(ax, f"FFT Analysis of I/O Bandwidth - {app}", "Frequency (Hz)", "Power")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(figdir / "fft" / f"{app}_fft.png", dpi=300)
            plt.close(fig)

            # Burst Detection
            detected_df, detect_meta = detect_adaptive(binned_df, window=args.window, fixed_percentile=args.fixed_percentile)
            detections_by_app[app] = detected_df
            detected_df.to_csv(csvdir / f"{app}_adaptive_detection_events.csv", index=False)
            
            detect_meta["application"] = app
            detection_meta_rows.append(detect_meta)

            fig, ax = plt.subplots(figsize=(6.8, 3.8))
            plot_detection_on_ax(ax, app, detected_df)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(figdir / "abpm" / f"{app}_ABPM.png", dpi=300)
            plt.close(fig)

        except Exception as e:
            print(f"ERROR: Failed to process {app}: {e}")
            import traceback
            traceback.print_exc()

    if not events_by_app:
        print("No CSV workloads were successfully processed. Exiting.")
        return

    # Compute and save aggregate tables
    sum_df, corr_df, ent_df, fft_df = compute_tables_csv(events_by_app)
    sum_df.to_csv(csvdir / "summary_metrics_from_csv_events.csv", index=False)
    corr_df.to_csv(csvdir / "correlation_metrics_from_csv_events.csv", index=False)
    ent_df.to_csv(csvdir / "entropy_metrics_from_csv_events.csv", index=False)
    fft_df.to_csv(csvdir / "dominant_frequency_metrics_from_csv_events.csv", index=False)
    
    if detection_meta_rows:
        pd.DataFrame(detection_meta_rows).to_csv(csvdir / "adaptive_detection_csv_summary.csv", index=False)

    # ---------------------------------------------------------------------------
    # Plot Combined Grid Figures (matching Fig1-5 of article)
    # ---------------------------------------------------------------------------
    apps_processed = sorted(events_by_app.keys())
    
    # Grid 1: Bandwidth Variation
    def bw_ax(ax, app):
        df = events_by_app[app]
        ax.plot(df["timestamp"], df["bandwidth_mb_s"], color="tab:blue", linewidth=0.8, alpha=0.75)
        setup_axes(ax, app, "Time (s)", "Bandwidth (MB/s)")
    combined_grid(apps_processed, bw_ax, "Bandwidth Variation of CSV HPC Workloads", figdir / "Fig1_CSV_Bandwidth_Variation.png")

    # Grid 2: combined CDF/PDF (row per workload, CDF + PDF columns)
    fig, axes = plt.subplots(len(apps_processed), 2, figsize=(10, 2.9 * len(apps_processed)))
    axes = np.asarray(axes).reshape(len(apps_processed), 2)
    for i, app in enumerate(apps_processed):
        x = events_by_app[app]["bandwidth_mb_s"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        xs = np.sort(x)
        cdf = np.arange(1, len(xs) + 1) / len(xs)
        axes[i, 0].plot(xs, cdf, color="tab:blue", linewidth=1.2)
        setup_axes(axes[i, 0], f"Bandwidth CDF of {app}", "Bandwidth (MB/s)", "CDF")
        axes[i, 1].hist(x, bins=30, color="tab:blue", alpha=0.9)
        setup_axes(axes[i, 1], f"Bandwidth PDF of {app}", "Bandwidth (MB/s)", "Frequency")
    fig.suptitle("I/O Bandwidth PDF and CDF of CSV HPC Workloads", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(figdir / "Fig2_CSV_PDF_CDF_of_HPC_Workloads.png", dpi=300)
    plt.close(fig)

    # Grid 3: ACF
    def acf_ax(ax, app):
        acf = autocorrelation(events_by_app[app]["bandwidth_mb_s"], 50)
        markerline, stemlines, baseline = ax.stem(acf["lag"], acf["acf"], basefmt=" ")
        style_acf_stem(ax, markerline, stemlines, baseline)
        ax.axhline(0, color="black", linewidth=0.8)
        setup_axes(ax, f"ACF Bandwidth of {app}", "Lag", "ACF")
    combined_grid(apps_processed, acf_ax, "ACF Bandwidth of CSV HPC Workloads", figdir / "Fig3_CSV_ACF_of_HPC_Workloads.png")

    # Grid 4: FFT
    def fft_ax(ax, app):
        spectrum, features = fft_spectrum(events_by_app[app])
        if not spectrum.empty:
            ax.plot(spectrum["frequency_hz"], spectrum["power"], color="tab:blue", linewidth=0.8)
            if math.isfinite(features["top_dominant_frequency_hz"]):
                ax.axvline(features["top_dominant_frequency_hz"], color="red", linestyle="--", linewidth=1.0)
        setup_axes(ax, f"FFT Analysis of Bandwidth - {app}", "Frequency (Hz)", "Power")
    combined_grid(apps_processed, fft_ax, "FFT Analysis of I/O Bandwidth for CSV HPC Workloads", figdir / "Fig4_CSV_FFT_Analysis.png")

    # Grid 5: Burst Detection
    if detections_by_app:
        def det_ax(ax, app):
            plot_detection_on_ax(ax, app, detections_by_app[app])
        combined_grid(apps_processed, det_ax, "Detected I/O Bursts for CSV HPC Workloads Using Adaptive Prediction Model", figdir / "Fig5_CSV_Detected_Bursts.png", ncols=2, figsize=(12, 4.2 * math.ceil(len(apps_processed)/2)))

    # Zip output files
    zip_path = args.output_dir.with_name("csv_reproduced_figures.zip")
    print(f"Creating zip bundle: {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(args.output_dir):
            for file in files:
                fpath = Path(root) / file
                arcname = fpath.relative_to(args.output_dir)
                zipf.write(fpath, arcname)

    print(f"Done. Output directory: {args.output_dir}")
    print(f"Zip bundle: {zip_path}")


if __name__ == "__main__":
    main()
