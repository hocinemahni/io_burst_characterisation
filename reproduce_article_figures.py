#!/usr/bin/env python3
"""I/O burst characterization and detection from Darshan traces.

Temporal detection uses timestamped DXT segments aggregated on a regular
time grid. Thresholds are causal and event output uses temporal persistence.
Controlled traces provide independent event labels; real-trace detections are
reported separately.
"""
from __future__ import annotations

import argparse
import math
import shutil
import struct
import time
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MB = 1024.0 * 1024.0
APP_ORDER = ["NAMD", "E3SM", "HACC", "IOR_HDF5", "LIFE-SCIENCE", "YOMBO"]

# Darshan POSIX / DXT parsing
POSIX_COUNTERS = """
POSIX_OPENS POSIX_FILENOS POSIX_DUPS POSIX_READS POSIX_WRITES POSIX_SEEKS
POSIX_STATS POSIX_MMAPS POSIX_FSYNCS POSIX_FDSYNCS POSIX_RENAME_SOURCES
POSIX_RENAME_TARGETS POSIX_RENAMED_FROM POSIX_MODE POSIX_BYTES_READ
POSIX_BYTES_WRITTEN POSIX_MAX_BYTE_READ POSIX_MAX_BYTE_WRITTEN POSIX_CONSEC_READS
POSIX_CONSEC_WRITES POSIX_SEQ_READS POSIX_SEQ_WRITES POSIX_RW_SWITCHES
POSIX_MEM_NOT_ALIGNED POSIX_MEM_ALIGNMENT POSIX_FILE_NOT_ALIGNED POSIX_FILE_ALIGNMENT
POSIX_MAX_READ_TIME_SIZE POSIX_MAX_WRITE_TIME_SIZE POSIX_SIZE_READ_0_100
POSIX_SIZE_READ_100_1K POSIX_SIZE_READ_1K_10K POSIX_SIZE_READ_10K_100K
POSIX_SIZE_READ_100K_1M POSIX_SIZE_READ_1M_4M POSIX_SIZE_READ_4M_10M
POSIX_SIZE_READ_10M_100M POSIX_SIZE_READ_100M_1G POSIX_SIZE_READ_1G_PLUS
POSIX_SIZE_WRITE_0_100 POSIX_SIZE_WRITE_100_1K POSIX_SIZE_WRITE_1K_10K
POSIX_SIZE_WRITE_10K_100K POSIX_SIZE_WRITE_100K_1M POSIX_SIZE_WRITE_1M_4M
POSIX_SIZE_WRITE_4M_10M POSIX_SIZE_WRITE_10M_100M POSIX_SIZE_WRITE_100M_1G
POSIX_SIZE_WRITE_1G_PLUS POSIX_STRIDE1_STRIDE POSIX_STRIDE2_STRIDE
POSIX_STRIDE3_STRIDE POSIX_STRIDE4_STRIDE POSIX_STRIDE1_COUNT POSIX_STRIDE2_COUNT
POSIX_STRIDE3_COUNT POSIX_STRIDE4_COUNT POSIX_ACCESS1_ACCESS POSIX_ACCESS2_ACCESS
POSIX_ACCESS3_ACCESS POSIX_ACCESS4_ACCESS POSIX_ACCESS1_COUNT POSIX_ACCESS2_COUNT
POSIX_ACCESS3_COUNT POSIX_ACCESS4_COUNT POSIX_FASTEST_RANK POSIX_FASTEST_RANK_BYTES
POSIX_SLOWEST_RANK POSIX_SLOWEST_RANK_BYTES
""".split()

POSIX_F_COUNTERS = """
POSIX_F_OPEN_START_TIMESTAMP POSIX_F_READ_START_TIMESTAMP POSIX_F_WRITE_START_TIMESTAMP
POSIX_F_CLOSE_START_TIMESTAMP POSIX_F_OPEN_END_TIMESTAMP POSIX_F_READ_END_TIMESTAMP
POSIX_F_WRITE_END_TIMESTAMP POSIX_F_CLOSE_END_TIMESTAMP POSIX_F_READ_TIME
POSIX_F_WRITE_TIME POSIX_F_META_TIME POSIX_F_MAX_READ_TIME POSIX_F_MAX_WRITE_TIME
POSIX_F_FASTEST_RANK_TIME POSIX_F_SLOWEST_RANK_TIME POSIX_F_VARIANCE_RANK_TIME
POSIX_F_VARIANCE_RANK_BYTES
""".split()

POSIX_RECORD_SIZE = 16 + len(POSIX_COUNTERS) * 8 + len(POSIX_F_COUNTERS) * 8
DXT_HEADER_SIZE = 16 + 8 + 64 + 8 + 8
DXT_SEGMENT_SIZE = 8 + 8 + 8 + 8


def infer_app_name(path: Path) -> str:
    name = path.name.lower()
    if "ior_hdf5" in name:
        return "IOR_HDF5"
    if "life-science" in name or "lifescience" in name:
        return "LIFE-SCIENCE"
    if "yombo" in name:
        return "YOMBO"
    if "namd" in name:
        return "NAMD"
    if "e3sm" in name:
        return "E3SM"
    if "hacc" in name:
        return "HACC"
    # ior_easy.darshan is a synthetic IOR trace and is excluded from real-trace analysis.
    if "ior_easy" in name:
        return "SYNTHETIC_IOR_LEGACY"
    if "ior" in name:
        return "IOR"
    return path.stem.split("_")[0].upper()


def natural_app_sort(apps: Iterable[str]) -> List[str]:
    rank = {a: i for i, a in enumerate(APP_ORDER)}
    return sorted(apps, key=lambda a: rank.get(a, 999))


def zlib_streams(blob: bytes) -> Iterable[Tuple[int, int, bytes]]:
    offsets: List[int] = []
    for magic in (b"\x78\x9c", b"\x78\xda", b"\x78\x01"):
        start = 0
        while True:
            idx = blob.find(magic, start)
            if idx < 0:
                break
            offsets.append(idx)
            start = idx + 1
    seen = set()
    for off in sorted(offsets):
        if off in seen:
            continue
        seen.add(off)
        try:
            obj = zlib.decompressobj()
            out = obj.decompress(blob[off:])
            consumed = len(blob[off:]) - len(obj.unused_data)
            if out:
                yield off, consumed, out
        except zlib.error:
            continue


def parse_posix_records(data: bytes) -> Optional[pd.DataFrame]:
    if len(data) == 0 or len(data) % POSIX_RECORD_SIZE != 0:
        return None
    rows = []
    for pos in range(0, len(data), POSIX_RECORD_SIZE):
        rec = data[pos:pos + POSIX_RECORD_SIZE]
        rid, rank = struct.unpack_from("<Qq", rec, 0)
        counters = struct.unpack_from("<" + "q" * len(POSIX_COUNTERS), rec, 16)
        fcounters = struct.unpack_from(
            "<" + "d" * len(POSIX_F_COUNTERS),
            rec,
            16 + len(POSIX_COUNTERS) * 8,
        )
        row = {"record_id": rid, "rank": rank}
        row.update(dict(zip(POSIX_COUNTERS, counters)))
        row.update(dict(zip(POSIX_F_COUNTERS, fcounters)))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    if (df[["POSIX_READS", "POSIX_WRITES", "POSIX_OPENS"]] < 0).any().any():
        return None
    if df["rank"].abs().max() > 10_000_000:
        return None
    total_bytes = df["POSIX_BYTES_READ"].sum() + df["POSIX_BYTES_WRITTEN"].sum()
    if total_bytes < 0 or total_bytes > 10**18:
        return None
    if (df["POSIX_READS"].sum() + df["POSIX_WRITES"].sum() + total_bytes) == 0:
        return None
    return df


def parse_dxt_records(data: bytes) -> Optional[pd.DataFrame]:
    pos = 0
    events = []
    while pos + DXT_HEADER_SIZE <= len(data):
        record_id, rank, shared_record = struct.unpack_from("<Qqq", data, pos)
        host_raw = data[pos + 24:pos + 88]
        hostname = host_raw.split(b"\0", 1)[0].decode("latin1", errors="ignore")
        write_count, read_count = struct.unpack_from("<qq", data, pos + 88)
        if not (0 <= write_count <= 10_000_000 and 0 <= read_count <= 10_000_000):
            return None
        rec_size = DXT_HEADER_SIZE + (write_count + read_count) * DXT_SEGMENT_SIZE
        if pos + rec_size > len(data):
            return None
        seg_pos = pos + DXT_HEADER_SIZE
        for io_type, count in (("write", write_count), ("read", read_count)):
            for _ in range(count):
                offset, length, start_time, end_time = struct.unpack_from("<qqdd", data, seg_pos)
                seg_pos += DXT_SEGMENT_SIZE
                if not (math.isfinite(start_time) and math.isfinite(end_time)):
                    continue
                if length < 0:
                    continue
                events.append({
                    "record_id": record_id,
                    "rank": rank,
                    "shared_record": shared_record,
                    "hostname": hostname,
                    "io_type": io_type,
                    "offset": offset,
                    "length_bytes": length,
                    "start_time": start_time,
                    "end_time": end_time,
                })
        pos += rec_size
    if pos != len(data) or not events:
        return None
    df = pd.DataFrame(events)
    if df["rank"].abs().max() > 10_000_000:
        return None
    return df


def read_darshan_log(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    blob = path.read_bytes()
    app = infer_app_name(path)
    version = blob[:8].split(b"\0", 1)[0].decode("ascii", errors="ignore")
    posix_frames: List[pd.DataFrame] = []
    dxt_frames: List[pd.DataFrame] = []
    n_streams = 0
    for off, _, out in zlib_streams(blob):
        n_streams += 1
        posix = parse_posix_records(out)
        if posix is not None:
            posix["source_offset"] = off
            posix_frames.append(posix)
        dxt = parse_dxt_records(out)
        if dxt is not None:
            dxt["source_offset"] = off
            dxt_frames.append(dxt)
    posix_df = pd.concat(posix_frames, ignore_index=True) if posix_frames else pd.DataFrame()
    dxt_df = pd.concat(dxt_frames, ignore_index=True) if dxt_frames else pd.DataFrame()
    if not posix_df.empty:
        posix_df["application"] = app
    if not dxt_df.empty:
        dxt_df["application"] = app
        dxt_df = dxt_df.drop_duplicates(
            subset=["record_id", "rank", "io_type", "offset", "length_bytes", "start_time", "end_time"]
        ).reset_index(drop=True)
    meta = {
        "application": app,
        "file": path.name,
        "version": version,
        "n_zlib_streams": n_streams,
        "n_posix_records": len(posix_df),
        "n_dxt_segments": len(dxt_df),
    }
    return posix_df, dxt_df, meta


@dataclass
class TraceRecord:
    app: str
    path: Path
    posix: pd.DataFrame
    dxt: pd.DataFrame
    meta: Dict[str, object]


def select_trace_per_app(paths: Sequence[Path]) -> Tuple[Dict[str, TraceRecord], pd.DataFrame]:
    """Select one run per application, preferring the run with the most DXT segments.

    Each application is represented by a single execution trace.
    """
    candidates: Dict[str, List[TraceRecord]] = {}
    all_meta = []
    for path in paths:
        posix, dxt, meta = read_darshan_log(path)
        app = str(meta["application"])
        all_meta.append(meta)
        if app not in APP_ORDER:
            continue
        candidates.setdefault(app, []).append(TraceRecord(app, path, posix, dxt, meta))

    selected: Dict[str, TraceRecord] = {}
    for app, recs in candidates.items():
        recs = sorted(
            recs,
            key=lambda r: (len(r.dxt), len(r.posix), r.path.name),
            reverse=True,
        )
        selected[app] = recs[0]
    return selected, pd.DataFrame(all_meta)


# Trace construction
def dxt_to_regular_bandwidth(dxt: pd.DataFrame, bin_width_s: float = 0.01) -> pd.DataFrame:
    """Aggregate DXT segments into a regular bandwidth time series.

    For a DXT segment i with interval [s_i, e_i] and B_i bytes, the contribution
    to bin j is B_i multiplied by the fraction of the segment duration that
    overlaps the bin. This assumes a constant transfer rate within each recorded
    DXT segment; no operation timestamps are invented.
    """
    if dxt is None or dxt.empty:
        return pd.DataFrame()
    bw = float(bin_width_s)
    if bw <= 0:
        raise ValueError("bin_width_s must be positive")

    d = dxt.copy()
    d["start_time"] = pd.to_numeric(d["start_time"], errors="coerce")
    d["end_time"] = pd.to_numeric(d["end_time"], errors="coerce")
    d["length_bytes"] = pd.to_numeric(d["length_bytes"], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["start_time", "end_time", "length_bytes"])
    d = d[d["length_bytes"] >= 0]
    if d.empty:
        return pd.DataFrame()

    t0 = float(d["start_time"].min())
    t1 = float(d["end_time"].max())
    if not (math.isfinite(t0) and math.isfinite(t1) and t1 >= t0):
        return pd.DataFrame()
    if t1 == t0:
        t1 = t0 + bw

    n_bins = int(math.ceil((t1 - t0) / bw)) + 1
    bytes_total = np.zeros(n_bins, dtype=float)
    bytes_read = np.zeros(n_bins, dtype=float)
    bytes_write = np.zeros(n_bins, dtype=float)
    op_mass = np.zeros(n_bins, dtype=float)

    for r in d.itertuples(index=False):
        start = float(r.start_time) - t0
        end = float(r.end_time) - t0
        length = float(r.length_bytes)
        if end <= start:
            end = start + 1e-12
        duration = end - start
        first = max(0, int(math.floor(start / bw)))
        last = min(n_bins - 1, int(math.floor((end - 1e-15) / bw)))
        if first == last:
            bytes_total[first] += length
            op_mass[first] += 1.0
            if r.io_type == "read":
                bytes_read[first] += length
            else:
                bytes_write[first] += length
            continue
        for j in range(first, last + 1):
            bs, be = j * bw, (j + 1) * bw
            overlap = max(0.0, min(end, be) - max(start, bs))
            if overlap <= 0:
                continue
            frac = overlap / duration
            contribution = length * frac
            bytes_total[j] += contribution
            op_mass[j] += frac
            if r.io_type == "read":
                bytes_read[j] += contribution
            else:
                bytes_write[j] += contribution

    ts = np.arange(n_bins, dtype=float) * bw
    out = pd.DataFrame({
        "timestamp_s": ts,
        "bandwidth_mb_s": bytes_total / MB / bw,
        "volume_mb": bytes_total / MB,
        "read_volume_mb": bytes_read / MB,
        "write_volume_mb": bytes_write / MB,
        "operation_mass": op_mass,
    })
    out["active"] = out["bandwidth_mb_s"] > 0
    return out


def posix_summary(app: str, posix: pd.DataFrame, dxt: pd.DataFrame, source_file: str) -> Dict[str, object]:
    row: Dict[str, object] = {
        "application": app,
        "source_file": source_file,
        "posix_records": int(len(posix)) if posix is not None else 0,
        "dxt_segments": int(len(dxt)) if dxt is not None else 0,
    }
    if posix is None or posix.empty:
        row.update({
            "write_operations": np.nan,
            "read_operations": np.nan,
            "write_volume_gib": np.nan,
            "read_volume_gib": np.nan,
        })
    else:
        row.update({
            "write_operations": int(posix["POSIX_WRITES"].sum()),
            "read_operations": int(posix["POSIX_READS"].sum()),
            "write_volume_gib": float(posix["POSIX_BYTES_WRITTEN"].sum() / (1024**3)),
            "read_volume_gib": float(posix["POSIX_BYTES_READ"].sum() / (1024**3)),
        })
    return row


def dxt_coverage_summary(app: str, posix: pd.DataFrame, dxt: pd.DataFrame) -> Dict[str, object]:
    """Compare DXT and POSIX byte totals for one selected run.

    A trace is eligible for temporal analysis when DXT is available and its
    total byte count differs from the POSIX total by no more than 5%.
    """
    pr = float(posix["POSIX_BYTES_READ"].sum()) if posix is not None and not posix.empty else np.nan
    pw = float(posix["POSIX_BYTES_WRITTEN"].sum()) if posix is not None and not posix.empty else np.nan
    dr = float(dxt.loc[dxt["io_type"] == "read", "length_bytes"].sum()) if dxt is not None and not dxt.empty else 0.0
    dw = float(dxt.loc[dxt["io_type"] == "write", "length_bytes"].sum()) if dxt is not None and not dxt.empty else 0.0
    pos_total = pr + pw if np.isfinite(pr) and np.isfinite(pw) else np.nan
    dxt_total = dr + dw
    ratio = dxt_total / pos_total if np.isfinite(pos_total) and pos_total > 0 else np.nan
    complete = bool(np.isfinite(ratio) and 0.95 <= ratio <= 1.05 and dxt_total > 0)
    return {
        "application": app,
        "posix_bytes": pos_total,
        "dxt_bytes": dxt_total,
        "dxt_posix_ratio": ratio,
        "dxt_complete": complete,
    }


# Statistical characterization
def normalized_histogram_entropy(values: pd.Series, bins: int = 16) -> float:
    """Distributional entropy; this is not interpreted as temporal unpredictability."""
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(arr) == 0:
        return np.nan
    counts, _ = np.histogram(arr, bins=bins)
    p = counts[counts > 0].astype(float)
    if p.size == 0:
        return np.nan
    p /= p.sum()
    h = -(p * np.log2(p)).sum()
    return float(h / np.log2(bins)) if bins > 1 else np.nan


def autocorrelation(values: pd.Series, max_lag: int = 100) -> pd.DataFrame:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    if len(x) < 2:
        return pd.DataFrame({"lag": [], "acf": []})
    x = x - x.mean()
    denom = float(np.dot(x, x))
    max_lag = min(int(max_lag), len(x) - 1)
    rows = []
    for lag in range(max_lag + 1):
        val = np.nan if denom == 0 else float(np.dot(x[:len(x)-lag], x[lag:]) / denom)
        rows.append({"lag": lag, "acf": val})
    return pd.DataFrame(rows)


def spectral_features(values: pd.Series, dt: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    if len(x) < 4 or np.std(x) <= 1e-15:
        return pd.DataFrame({"frequency_hz": [], "power": []}), {
            "dominant_frequency_hz": np.nan,
            "dominant_period_s": np.nan,
            "spectral_concentration": 0.0,
        }
    y = np.log1p(np.maximum(x, 0.0))
    y = y - y.mean()
    vals = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=dt)
    power = np.abs(vals) ** 2
    mask = freqs > 0
    freqs, power = freqs[mask], power[mask]
    if len(power) == 0 or power.sum() <= 0:
        return pd.DataFrame({"frequency_hz": [], "power": []}), {
            "dominant_frequency_hz": np.nan,
            "dominant_period_s": np.nan,
            "spectral_concentration": 0.0,
        }
    idx = int(np.argmax(power))
    f = float(freqs[idx])
    concentration = float(power[idx] / power.sum())
    return pd.DataFrame({"frequency_hz": freqs, "power": power}), {
        "dominant_frequency_hz": f,
        "dominant_period_s": float(1.0 / f) if f > 0 else np.nan,
        "spectral_concentration": concentration,
    }


def characterize_signal(app: str, s: pd.DataFrame, bin_width_s: float) -> Dict[str, object]:
    x = s["bandwidth_mb_s"].astype(float)
    _, spec = spectral_features(x, bin_width_s)
    return {
        "application": app,
        "duration_s": float(s["timestamp_s"].iloc[-1]) if len(s) else 0.0,
        "n_bins": int(len(s)),
        "bin_width_s": float(bin_width_s),
        "mean_bandwidth_mb_s": float(x.mean()),
        "std_bandwidth_mb_s": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "cv_bandwidth": float(x.std(ddof=1) / x.mean()) if x.mean() > 0 and len(x) > 1 else np.nan,
        "p95_bandwidth_mb_s": float(x.quantile(0.95)),
        "max_bandwidth_mb_s": float(x.max()),
        "active_fraction": float((x > 0).mean()),
        "distributional_entropy": normalized_histogram_entropy(np.log1p(x)),
        **spec,
    }


# Causal adaptive burst detector
def spectral_concentration_array(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if len(y) < 16 or np.std(y) <= 1e-15:
        return 0.0
    y = y - y.mean()
    p = np.abs(np.fft.rfft(y)) ** 2
    if len(p) <= 1:
        return 0.0
    p = p[1:]
    total = float(p.sum())
    return float(p.max() / total) if total > 0 else 0.0


def causal_robust_threshold(x: pd.Series, window: int, tau: float, quantile_floor: float) -> pd.Series:
    """Robust causal threshold using log bandwidth and IQR scale."""
    y = np.log1p(x.astype(float).clip(lower=0))
    h = y.shift(1)
    med = h.rolling(window, min_periods=window).median()
    q25 = h.rolling(window, min_periods=window).quantile(0.25)
    q75 = h.rolling(window, min_periods=window).quantile(0.75)
    # IQR / 1.349 estimates sigma under a Gaussian model but remains robust to outliers.
    scale = ((q75 - q25) / 1.349).clip(lower=1e-6)
    t_robust = np.expm1(med + tau * scale)
    q_floor = x.shift(1).rolling(window, min_periods=window).quantile(quantile_floor)
    return pd.concat([t_robust, q_floor], axis=1).max(axis=1)


def causal_mean_threshold(x: pd.Series, window: int, k: float) -> pd.Series:
    h = x.shift(1)
    mu = h.rolling(window, min_periods=window).mean()
    sigma = h.rolling(window, min_periods=window).std(ddof=1)
    return mu + k * sigma


def causal_quantile_threshold(x: pd.Series, window: int, q: float) -> pd.Series:
    return x.shift(1).rolling(window, min_periods=window).quantile(q)


def causal_hampel_threshold(x: pd.Series, window: int, z: float = 3.0) -> pd.Series:
    """Causal Hampel threshold on log-bandwidth using median absolute deviation.

    The Gaussian-consistent factor 1.4826 is used only as a scale calibration;
    the estimator itself remains robust to isolated extreme samples.
    """
    y = np.log1p(x.astype(float).clip(lower=0)).to_numpy(float)
    out = np.full(len(y), np.nan, dtype=float)
    if len(y) <= window:
        return pd.Series(out, index=x.index)
    from numpy.lib.stride_tricks import sliding_window_view
    hist = sliding_window_view(y, window)[:-1]
    med = np.median(hist, axis=1)
    mad = np.median(np.abs(hist - med[:, None]), axis=1)
    scale = np.maximum(1.4826 * mad, 1e-6)
    out[window:] = np.expm1(med + z * scale)
    return pd.Series(out, index=x.index)


def causal_persistence(candidates: np.ndarray, window: int, hits: int) -> np.ndarray:
    s = pd.Series(np.asarray(candidates, dtype=bool).astype(int))
    return (s.rolling(window, min_periods=window).sum() >= hits).fillna(False).to_numpy(bool)


def run_crad(
    signal: pd.DataFrame,
    window: int,
    tau: float = 3.0,
    robust_quantile_floor: float = 0.90,
    mean_k: float = 3.0,
    active_threshold: float = 0.20,
    periodic_threshold: float = 0.25,
    periodic_update: int = 20,
    persistence_window: int = 5,
    persistence_hits: int = 3,
) -> pd.DataFrame:
    out = signal.copy().reset_index(drop=True)
    x = out["bandwidth_mb_s"].astype(float)
    if len(out) < window + persistence_window:
        out["threshold_robust"] = np.nan
        out["threshold_mean"] = np.nan
        out["spectral_concentration_local"] = np.nan
        out["active_fraction_local"] = np.nan
        out["threshold_crad"] = np.nan
        out["branch"] = "insufficient_history"
        out["candidate"] = False
        out["burst"] = False
        return out

    t_robust = causal_robust_threshold(x, window, tau, robust_quantile_floor)
    t_mean = causal_mean_threshold(x, window, mean_k)
    active_fraction = (x.shift(1) > 0).rolling(window, min_periods=window).mean()

    logx = np.log1p(x.clip(lower=0).to_numpy(float))
    pscore = np.full(len(out), np.nan, dtype=float)
    last = 0.0
    step = max(1, int(periodic_update))
    for t in range(window, len(out), step):
        last = spectral_concentration_array(logx[t-window:t])
        pscore[t:min(t + step, len(out))] = last

    use_mean = (active_fraction.to_numpy() < active_threshold) | (pscore > periodic_threshold)
    t_final = np.where(use_mean, t_mean.to_numpy(), t_robust.to_numpy())
    candidate = x.to_numpy() > t_final
    candidate[np.isnan(t_final)] = False
    burst = causal_persistence(candidate, persistence_window, persistence_hits)

    out["threshold_robust"] = t_robust
    out["threshold_mean"] = t_mean
    out["spectral_concentration_local"] = pscore
    out["active_fraction_local"] = active_fraction
    out["threshold_crad"] = t_final
    out["branch"] = np.where(use_mean, "mean", "robust")
    out.loc[np.isnan(t_final), "branch"] = "warmup"
    out["candidate"] = candidate
    out["burst"] = burst
    return out


def run_baseline(signal: pd.DataFrame, method: str, window: int, persistence_window: int, persistence_hits: int) -> np.ndarray:
    x = signal["bandwidth_mb_s"].astype(float)
    if method == "global_p95":
        threshold = pd.Series(float(x.quantile(0.95)), index=x.index)
    elif method == "causal_mean_3sigma":
        threshold = causal_mean_threshold(x, window, 3.0)
    elif method == "causal_p99":
        threshold = causal_quantile_threshold(x, window, 0.99)
    elif method == "robust_only":
        threshold = causal_robust_threshold(x, window, 3.0, 0.90)
    elif method == "causal_hampel":
        threshold = causal_hampel_threshold(x, window, 3.0)
    else:
        raise ValueError(method)
    c = (x > threshold).fillna(False).to_numpy(bool)
    return causal_persistence(c, persistence_window, persistence_hits)


def mask_to_events(mask: np.ndarray, max_gap_bins: int = 1) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(idx) == 0:
        return []
    events: List[Tuple[int, int]] = []
    start = prev = int(idx[0])
    for raw in idx[1:]:
        i = int(raw)
        if i - prev <= max_gap_bins + 1:
            prev = i
        else:
            events.append((start, prev))
            start = prev = i
    events.append((start, prev))
    return events


def event_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Temporal intersection-over-union for two inclusive event intervals."""
    s1, e1 = a
    s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    if inter == 0:
        return 0.0
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
    return inter / union


def event_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    min_iou: float = 0.30,
    max_gap_bins: int = 1,
) -> Dict[str, float]:
    """One-to-one event metrics using descending temporal IoU matching."""
    pred_events = mask_to_events(pred, max_gap_bins=max_gap_bins)
    true_events = mask_to_events(truth, max_gap_bins=0)

    pairs = []
    for gi, gt in enumerate(true_events):
        for pi, pr in enumerate(pred_events):
            iou = event_iou(gt, pr)
            if iou >= min_iou:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)

    used_gt, used_pr = set(), set()
    delays, duration_errors, matched_ious = [], [], []
    for iou, gi, pi in pairs:
        if gi in used_gt or pi in used_pr:
            continue
        used_gt.add(gi)
        used_pr.add(pi)
        gs, ge = true_events[gi]
        ps, pe = pred_events[pi]
        delays.append(ps - gs)
        duration_errors.append(abs((pe - ps + 1) - (ge - gs + 1)))
        matched_ious.append(iou)

    tp = len(used_gt)
    fp = len(pred_events) - tp
    fn = len(true_events) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_events": tp,
        "fp_events": fp,
        "fn_events": fn,
        "mean_detection_delay_bins": float(np.mean(delays)) if delays else np.nan,
        "mean_abs_duration_error_bins": float(np.mean(duration_errors)) if duration_errors else np.nan,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else np.nan,
        "predicted_events": len(pred_events),
        "true_events": len(true_events),
    }


# Controlled benchmark
def generate_synthetic_scenario(kind: str, n: int = 5000, seed: int = 0, dt: float = 0.01) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    if kind == "stationary":
        e = rng.normal(0.0, 0.18, n)
        for i in range(1, n):
            e[i] = 0.7 * e[i-1] + e[i]
        x = 100.0 * np.exp(e)
    elif kind == "nonstationary":
        levels = np.array([60.0, 180.0, 90.0, 250.0, 120.0])
        seg = n // len(levels)
        e = rng.normal(0.0, 0.15, n)
        for i in range(1, n):
            e[i] = 0.6 * e[i-1] + e[i]
        x = np.empty(n, dtype=float)
        for j, level in enumerate(levels):
            a = j * seg
            b = n if j == len(levels) - 1 else (j + 1) * seg
            x[a:b] = level * np.exp(e[a:b])
    elif kind == "periodic":
        t = np.arange(n)
        baseline = 80.0 + 100.0 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 250.0))
        x = baseline * np.exp(rng.normal(0.0, 0.12, n))
    elif kind == "sparse":
        x = np.zeros(n, dtype=float)
        active = rng.random(n) < 0.08
        x[active] = rng.lognormal(mean=np.log(20.0), sigma=0.5, size=int(active.sum()))
        for st in rng.integers(200, n - 20, 20):
            x[st:st+5] += rng.lognormal(np.log(25.0), 0.3, 5)
    else:
        raise ValueError(f"Unknown synthetic scenario: {kind}")

    truth = np.zeros(n, dtype=bool)
    starts: List[int] = []
    while len(starts) < 20:
        st = int(rng.integers(300, n - 20))
        if any(abs(st - s) < 40 for s in starts):
            continue
        if kind == "nonstationary":
            boundaries = [j * (n // 5) for j in range(1, 5)]
            if min(abs(st - b) for b in boundaries) < 100:
                continue
        starts.append(st)
        duration = int(rng.integers(3, 11))
        local = x[max(0, st - 200):st]
        positive = local[local > 0]
        base = float(np.median(positive)) if len(positive) else 10.0
        amplitude = base * float(rng.uniform(2.5, 5.0))
        shape = np.hanning(duration + 2)[1:-1] if duration > 2 else np.ones(duration)
        shape = 0.5 + 0.5 * shape
        x[st:st+duration] += amplitude * shape
        truth[st:st+duration] = True

    return pd.DataFrame({"timestamp_s": np.arange(n) * dt, "bandwidth_mb_s": x}), truth



def generate_background(kind: str, n: int, seed: int) -> np.ndarray:
    """Generate a burst-free background for independent baseline calibration."""
    rng = np.random.default_rng(seed)
    if kind == "stationary":
        e = rng.normal(0.0, 0.18, n)
        for i in range(1, n):
            e[i] = 0.7 * e[i-1] + e[i]
        return 100.0 * np.exp(e)
    if kind == "nonstationary":
        levels = np.array([60.0, 180.0, 90.0, 250.0, 120.0])
        seg = n // len(levels)
        e = rng.normal(0.0, 0.15, n)
        for i in range(1, n):
            e[i] = 0.6 * e[i-1] + e[i]
        x = np.empty(n, dtype=float)
        for j, level in enumerate(levels):
            a = j * seg
            b = n if j == len(levels) - 1 else (j + 1) * seg
            x[a:b] = level * np.exp(e[a:b])
        return x
    if kind == "periodic":
        t = np.arange(n)
        baseline = 80.0 + 100.0 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 250.0))
        return baseline * np.exp(rng.normal(0.0, 0.12, n))
    if kind == "sparse":
        x = np.zeros(n, dtype=float)
        active = rng.random(n) < 0.08
        x[active] = rng.lognormal(mean=np.log(20.0), sigma=0.5, size=int(active.sum()))
        for st in rng.integers(200, n - 20, 20):
            x[st:st+5] += rng.lognormal(np.log(25.0), 0.3, 5)
        return x
    raise ValueError(f"Unknown synthetic scenario: {kind}")


def calibrated_mu_k_sigma_mask(
    signal: pd.DataFrame,
    calibration_signal: pd.DataFrame,
    persistence_window: int = 5,
    persistence_hits: int = 3,
    target_tail: float = 0.01,
) -> Tuple[np.ndarray, float, float]:
    """Calibrate k on an independent burst-free background, then freeze it on test data."""
    cal = calibration_signal["bandwidth_mb_s"].astype(float).to_numpy()
    mu = float(np.mean(cal))
    sigma = float(np.std(cal, ddof=1))
    q = float(np.quantile(cal, 1.0 - target_tail))
    k = max(0.0, (q - mu) / sigma) if sigma > 1e-12 else 0.0
    threshold = mu + k * sigma
    x = signal["bandwidth_mb_s"].astype(float).to_numpy()
    candidate = x > threshold
    return causal_persistence(candidate, persistence_window, persistence_hits), k, threshold

def evaluate_synthetic(
    seeds: int,
    bin_width_s: float,
    window: int,
    tau: float,
    periodic_threshold: float,
    active_threshold: float,
    periodic_update: int,
    persistence_window: int,
    persistence_hits: int,
) -> pd.DataFrame:
    rows = []
    methods = ["global_p95", "calibrated_mu_k_sigma", "causal_mean_3sigma", "causal_hampel", "crad"]
    for scenario in ["stationary", "nonstationary", "periodic", "sparse"]:
        for seed in range(seeds):
            signal, truth = generate_synthetic_scenario(scenario, seed=seed, dt=bin_width_s)
            calibration = None
            for method in methods:
                extra = {}
                if method == "crad":
                    det = run_crad(
                        signal, window=window, tau=tau,
                        active_threshold=active_threshold,
                        periodic_threshold=periodic_threshold,
                        periodic_update=periodic_update,
                        persistence_window=persistence_window,
                        persistence_hits=persistence_hits,
                    )["burst"].to_numpy(bool)
                elif method == "calibrated_mu_k_sigma":
                    if calibration is None:
                        bg = generate_background(scenario, len(signal), seed + 100000)
                        calibration = pd.DataFrame({
                            "timestamp_s": np.arange(len(bg)) * bin_width_s,
                            "bandwidth_mb_s": bg,
                        })
                    det, k, threshold = calibrated_mu_k_sigma_mask(
                        signal, calibration, persistence_window, persistence_hits
                    )
                    extra = {"calibrated_k": k, "calibrated_threshold": threshold}
                else:
                    det = run_baseline(signal, method, window, persistence_window, persistence_hits)
                m = event_metrics(det, truth, min_iou=0.30)
                rows.append({"scenario": scenario, "seed": seed, "method": method, **m, **extra})
    return pd.DataFrame(rows)

def sensitivity_analysis(
    seeds: int,
    bin_width_s: float,
    persistence_window: int,
    persistence_hits: int,
    seed_offset: int = 1000,
) -> pd.DataFrame:
    """One-factor-at-a-time sensitivity on a seed range disjoint from evaluation."""
    configs = []
    # History-window sweep.
    configs += [(w, 3.0, 0.25) for w in [0.5, 1.0, 2.0, 4.0, 8.0, 15.0]]
    # Threshold-multiplier sweep.
    configs += [(2.0, tau, 0.25) for tau in [3.0, 3.5, 4.0]]
    # Spectral-concentration threshold sweep.
    configs += [(2.0, 3.0, cmin) for cmin in [0.20, 0.25, 0.30]]
    configs = list(dict.fromkeys(configs))

    rows = []
    for history_s, tau, pthr in configs:
        window = max(20, int(round(history_s / bin_width_s)))
        f1s = []
        for scenario in ["stationary", "nonstationary", "periodic", "sparse"]:
            for seed in range(seed_offset, seed_offset + seeds):
                signal, truth = generate_synthetic_scenario(scenario, seed=seed, dt=bin_width_s)
                det = run_crad(
                    signal, window=window, tau=tau,
                    periodic_threshold=pthr,
                    persistence_window=persistence_window,
                    persistence_hits=persistence_hits,
                )["burst"].to_numpy(bool)
                f1s.append(event_metrics(det, truth, min_iou=0.30)["f1"])
        f1 = np.asarray(f1s, dtype=float)
        rows.append({
            "history_s": history_s,
            "window_bins": window,
            "tau": tau,
            "periodic_threshold": pthr,
            "macro_event_f1": float(np.mean(f1)),
            "standard_error": float(np.std(f1, ddof=1) / np.sqrt(len(f1))) if len(f1) > 1 else np.nan,
            "n_scenario_seed_pairs": int(len(f1)),
        })
    return pd.DataFrame(rows)


# Output helpers
def ensure_dirs(out: Path) -> Dict[str, Path]:
    dirs = {
        "csv": out / "csv",
        "figures": out / "figures",
        "tables": out / "tables",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_")


def write_latex_tables(
    outdir: Path,
    workload: pd.DataFrame,
    synth_summary: pd.DataFrame,
    real_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    # One table row per selected workload run.
    cols = ["application", "read_operations", "write_operations", "read_volume_gib", "write_volume_gib", "dxt_segments"]
    lines = [
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Workload & Reads & Writes & Read GiB & Write GiB & DXT seg. \\", r"\midrule",
    ]
    for _, r in workload[cols].iterrows():
        def iv(v): return "--" if pd.isna(v) else f"{int(v):,}"
        def fv(v): return "--" if pd.isna(v) else f"{float(v):.3g}"
        lines.append(f"{latex_escape(r.application)} & {iv(r.read_operations)} & {iv(r.write_operations)} & {fv(r.read_volume_gib)} & {fv(r.write_volume_gib)} & {iv(r.dxt_segments)}" + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "workload_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Event-level F1 table for the primary IoU criterion.
    pretty = {
        "global_p95": "P95",
        "calibrated_mu_k_sigma": r"Calibrated $\mu+k\sigma$",
        "causal_mean_3sigma": r"Rolling $\mu+3\sigma$",
        "causal_hampel": "Median/MAD",
        "crad": r"\textbf{Proposed}",
    }
    methods = ["global_p95", "calibrated_mu_k_sigma", "causal_mean_3sigma", "causal_hampel", "crad"]
    pivot = synth_summary.pivot(index="method", columns="scenario", values="f1_mean")
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Method & Stationary & Non-stat. & Periodic & Sparse & Mean \\", r"\midrule"]
    for method in methods:
        vals = [float(pivot.loc[method, sc]) for sc in ["stationary", "nonstationary", "periodic", "sparse"]]
        lines.append(f"{pretty[method]} & " + " & ".join(f"{v:.3f}" for v in vals) + f" & {np.mean(vals):.3f}" + " \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "synthetic_f1.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Real-trace detection summary.
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Workload & Duration (s) & Bins & Events & Burst bins (\%) \\", r"\midrule"]
    for _, r in real_summary.iterrows():
        lines.append(f"{latex_escape(r.application)} & {r.duration_s:.2f} & {int(r.n_bins):,} & {int(r.detected_events)} & {100*r.burst_bin_fraction:.2f}" + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "real_detection_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Sensitivity summary.
    best = sensitivity.sort_values("macro_event_f1", ascending=False).iloc[0]
    chosen = sensitivity[(sensitivity["history_s"] == 2.0) & (sensitivity["tau"] == 3.0) & (sensitivity["periodic_threshold"] == 0.25)].iloc[0]
    text = "\n".join([
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Configuration & History (s) & $\tau$ & $P_{thr}$ & Macro F1 \\",
        r"\midrule",
        f"Chosen & {chosen.history_s:.1f} & {chosen.tau:.1f} & {chosen.periodic_threshold:.2f} & {chosen.macro_event_f1:.3f}" + " \\\\",
        f"Best tested point & {best.history_s:.1f} & {best.tau:.1f} & {best.periodic_threshold:.2f} & {best.macro_event_f1:.3f}" + " \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n"
    (outdir / "sensitivity_summary.tex").write_text(text, encoding="utf-8")


def plot_real_detection(app: str, detected: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    ax.plot(detected["timestamp_s"], detected["bandwidth_mb_s"], linewidth=0.8, label="Bandwidth")
    ax.plot(detected["timestamp_s"], detected["threshold_crad"], linewidth=1.0, linestyle="--", label="Local threshold")

    candidate = detected["candidate"].to_numpy(bool)
    if candidate.any():
        ax.scatter(
            detected.loc[candidate, "timestamp_s"],
            detected.loc[candidate, "bandwidth_mb_s"],
            s=13, marker="x", linewidths=0.8, label="Threshold crossing", zorder=4,
        )

    events = mask_to_events(detected["burst"].to_numpy(bool), max_gap_bins=1)
    first = True
    for start_i, end_i in events:
        ax.axvspan(
            detected["timestamp_s"].iloc[start_i],
            detected["timestamp_s"].iloc[end_i],
            alpha=0.15, label="Declared event" if first else None, zorder=0,
        )
        first = False

    ax.set_title(app)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bandwidth (MiB/s)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

def plot_synthetic_f1(summary: pd.DataFrame, outpath: Path) -> None:
    methods = ["global_p95", "calibrated_mu_k_sigma", "causal_mean_3sigma", "causal_hampel", "crad"]
    labels = ["P95", "Calibrated mu+k sigma", "Rolling mu+3 sigma", "Median/MAD", "Proposed"]
    scenarios = ["stationary", "nonstationary", "periodic", "sparse"]
    x = np.arange(len(scenarios))
    width = 0.15
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for i, (method, label) in enumerate(zip(methods, labels)):
        vals = [float(summary[(summary.method == method) & (summary.scenario == sc)].f1_mean.iloc[0]) for sc in scenarios]
        ax.bar(x + (i - (len(methods)-1)/2) * width, vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["Stationary", "Non-stationary", "Periodic", "Sparse"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Event-level F1 (IoU >= 0.30)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

def plot_sensitivity(sens: pd.DataFrame, outpath: Path) -> None:
    d = sens[(np.isclose(sens["periodic_threshold"], 0.25)) & (np.isclose(sens["tau"], 3.0))].copy()
    d = d.sort_values("history_s")
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    yerr = 1.96 * d["standard_error"].to_numpy(float) if "standard_error" in d else None
    ax.errorbar(d.history_s, d.macro_event_f1, yerr=yerr, marker="o", capsize=3)
    ax.set_xlabel("History length (s)")
    ax.set_ylabel("Macro event F1 (IoU >= 0.30)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# Command-line entry point
def main() -> None:
    p = argparse.ArgumentParser(description="Rigorous causal I/O burst analysis from Darshan traces")
    p.add_argument("--input-dir", type=Path, default=Path("logs"))
    p.add_argument("--output-dir", type=Path, default=Path("results_rigorous"))
    p.add_argument("--bin-width", type=float, default=0.01, help="Regular DXT bin width in seconds")
    p.add_argument("--history-seconds", type=float, default=2.0)
    p.add_argument("--tau", type=float, default=3.0)
    p.add_argument("--active-threshold", type=float, default=0.20)
    p.add_argument("--periodic-threshold", type=float, default=0.25)
    p.add_argument("--periodic-update", type=int, default=20)
    p.add_argument("--persistence-window", type=int, default=5)
    p.add_argument("--persistence-hits", type=int, default=3)
    p.add_argument("--synthetic-seeds", type=int, default=30)
    p.add_argument("--sensitivity-seeds", type=int, default=30)
    p.add_argument("--sensitivity-seed-offset", type=int, default=1000)
    p.add_argument("--skip-sensitivity", action="store_true")
    p.add_argument("--clean", action="store_true")
    args = p.parse_args()

    if args.clean and args.output_dir.exists():
        resolved = args.output_dir.resolve()
        cwd = Path.cwd().resolve()
        home = Path.home().resolve()
        protected = {Path("/").resolve(), cwd, home}
        if resolved in protected or resolved in cwd.parents or resolved in home.parents:
            raise SystemExit(f"Refusing to delete protected output directory: {resolved}")
        shutil.rmtree(resolved)
    dirs = ensure_dirs(args.output_dir)

    log_paths = sorted(args.input_dir.glob("*.darshan"))
    if not log_paths:
        raise SystemExit(f"No Darshan logs in {args.input_dir}")

    selected, all_meta = select_trace_per_app(log_paths)
    all_meta.to_csv(dirs["csv"] / "all_parse_candidates.csv", index=False)

    selection_rows = []
    workload_rows = []
    signal_rows = []
    coverage_rows = []
    signals: Dict[str, pd.DataFrame] = {}
    for app in natural_app_sort(selected.keys()):
        rec = selected[app]
        selection_rows.append({
            "application": app,
            "selected_file": rec.path.name,
            "selected_dxt_segments": len(rec.dxt),
            "selected_posix_records": len(rec.posix),
        })
        workload_rows.append(posix_summary(app, rec.posix, rec.dxt, rec.path.name))
        coverage_rows.append(dxt_coverage_summary(app, rec.posix, rec.dxt))
        if not rec.dxt.empty:
            s = dxt_to_regular_bandwidth(rec.dxt, args.bin_width)
            signals[app] = s
            s.to_csv(dirs["csv"] / f"{app}_regular_bandwidth.csv", index=False)
            signal_rows.append(characterize_signal(app, s, args.bin_width))

    selection = pd.DataFrame(selection_rows)
    workload = pd.DataFrame(workload_rows)
    characterization = pd.DataFrame(signal_rows)
    coverage = pd.DataFrame(coverage_rows)
    selection.to_csv(dirs["csv"] / "selected_runs.csv", index=False)
    workload.to_csv(dirs["csv"] / "workload_summary.csv", index=False)
    characterization.to_csv(dirs["csv"] / "dxt_characterization.csv", index=False)
    coverage.to_csv(dirs["csv"] / "dxt_coverage.csv", index=False)

    window = max(20, int(round(args.history_seconds / args.bin_width)))
    real_rows = []
    for app in natural_app_sort(signals.keys()):
        s = signals[app]
        cov_row = coverage[coverage.application == app].iloc[0]
        if not bool(cov_row.dxt_complete):
            real_rows.append({
                "application": app,
                "duration_s": float(s.timestamp_s.iloc[-1]),
                "n_bins": len(s),
                "detected_events": 0,
                "burst_bin_fraction": 0.0,
                "runtime_ms": np.nan,
                "microseconds_per_bin": np.nan,
                "mean_branch_fraction": np.nan,
                "status": "excluded: incomplete or duplicated DXT coverage",
            })
            continue
        if len(s) < window + args.persistence_window:
            real_rows.append({
                "application": app,
                "duration_s": float(s.timestamp_s.iloc[-1]),
                "n_bins": len(s),
                "detected_events": 0,
                "burst_bin_fraction": 0.0,
                "runtime_ms": np.nan,
                "microseconds_per_bin": np.nan,
                "mean_branch_fraction": np.nan,
                "status": "excluded: insufficient history",
            })
            continue
        t0 = time.perf_counter()
        d = run_crad(
            s, window=window, tau=args.tau,
            active_threshold=args.active_threshold,
            periodic_threshold=args.periodic_threshold,
            periodic_update=args.periodic_update,
            persistence_window=args.persistence_window,
            persistence_hits=args.persistence_hits,
        )
        runtime = time.perf_counter() - t0
        d.to_csv(dirs["csv"] / f"{app}_crad_detection.csv", index=False)
        events = mask_to_events(d["burst"].to_numpy(bool), max_gap_bins=1)
        real_rows.append({
            "application": app,
            "duration_s": float(s.timestamp_s.iloc[-1]),
            "n_bins": len(s),
            "detected_events": len(events),
            "burst_bin_fraction": float(d["burst"].mean()),
            "runtime_ms": 1000.0 * runtime,
            "microseconds_per_bin": 1e6 * runtime / len(s),
            "mean_branch_fraction": float((d["branch"] == "mean").mean()),
            "status": "analyzed",
        })
        plot_real_detection(app, d, dirs["figures"] / f"{app}_crad_detection.png")

    real_summary = pd.DataFrame(real_rows)
    real_summary.to_csv(dirs["csv"] / "real_detection_summary.csv", index=False)

    synth = evaluate_synthetic(
        seeds=args.synthetic_seeds,
        bin_width_s=args.bin_width,
        window=window,
        tau=args.tau,
        periodic_threshold=args.periodic_threshold,
        active_threshold=args.active_threshold,
        periodic_update=args.periodic_update,
        persistence_window=args.persistence_window,
        persistence_hits=args.persistence_hits,
    )
    synth.to_csv(dirs["csv"] / "synthetic_event_metrics_all_runs.csv", index=False)
    synth_summary = synth.groupby(["scenario", "method"], as_index=False).agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        delay_bins_mean=("mean_detection_delay_bins", "mean"),
    )
    synth_summary.to_csv(dirs["csv"] / "synthetic_event_metrics_summary.csv", index=False)
    plot_synthetic_f1(synth_summary, dirs["figures"] / "synthetic_event_f1.png")

    if args.skip_sensitivity:
        sensitivity = pd.DataFrame([{
            "history_s": args.history_seconds,
            "window_bins": window,
            "tau": args.tau,
            "periodic_threshold": args.periodic_threshold,
            "macro_event_f1": float(synth[synth.method == "crad"].f1.mean()),
        }])
    else:
        sensitivity = sensitivity_analysis(
            seeds=args.sensitivity_seeds,
            bin_width_s=args.bin_width,
            persistence_window=args.persistence_window,
            persistence_hits=args.persistence_hits,
            seed_offset=args.sensitivity_seed_offset,
        )
        sensitivity.to_csv(dirs["csv"] / "crad_sensitivity.csv", index=False)
        plot_sensitivity(sensitivity, dirs["figures"] / "crad_sensitivity.png")

    # Keep only traces accepted by the temporal coverage check.
    real_for_table = real_summary[real_summary.status == "analyzed"].copy()
    write_latex_tables(dirs["tables"], workload, synth_summary, real_for_table, sensitivity)

    # Save the detector configuration in machine-readable form.
    config = pd.DataFrame([{
        "bin_width_s": args.bin_width,
        "history_seconds": args.history_seconds,
        "window_bins": window,
        "tau": args.tau,
        "robust_quantile_floor": 0.90,
        "mean_k": 3.0,
        "active_threshold": args.active_threshold,
        "periodic_threshold": args.periodic_threshold,
        "periodic_update_bins": args.periodic_update,
        "persistence_window_bins": args.persistence_window,
        "persistence_hits": args.persistence_hits,
        "synthetic_seeds": args.synthetic_seeds,
        "sensitivity_seeds": args.sensitivity_seeds,
        "sensitivity_seed_offset": args.sensitivity_seed_offset,
    }])
    config.to_csv(dirs["csv"] / "analysis_config.csv", index=False)

    notes = f"""# Reproduction notes

- Real temporal analysis uses **DXT only**.
- POSIX-only traces are used for aggregate characterization, not temporal detection.
- `ior_easy.darshan` is treated as a synthetic IOR trace and excluded from real-trace analysis.
- DXT bin width: {args.bin_width:g} s.
- Causal history: {args.history_seconds:g} s ({window} bins).
- CRAD tau: {args.tau:g}; active threshold: {args.active_threshold:g}; periodic threshold: {args.periodic_threshold:g}.
- Persistence: {args.persistence_hits} of the last {args.persistence_window} bins.
- Synthetic evaluation seeds: {args.synthetic_seeds} per scenario.
- Sensitivity analysis uses a disjoint seed range starting at {args.sensitivity_seed_offset}.
- Real temporal detections are reported only when DXT/POSIX byte coverage is within 5%.
- Real-trace detections are descriptive because independent system-impact labels are unavailable.
- Parser outputs can be cross-checked with the official Darshan/PyDarshan toolchain.
"""
    (args.output_dir / "REPRODUCTION_NOTES.md").write_text(notes, encoding="utf-8")
    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
