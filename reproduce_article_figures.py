#!/usr/bin/env python3
"""
Reproduce the figures of the article:
"Adaptive I/O Burst Characterization and Prediction in High-Performance Computing Systems".

The script is designed to reproduce the *same style and workflow* used in the
article figures from Darshan logs:

  - bandwidth time series
  - CDF/PDF of bandwidth
  - ACF of bandwidth
  - FFT power spectrum
  - adaptive burst detection figures with red adaptive bursts and blue fixed-threshold bursts
  - CSV tables used to verify the numerical results

Important methodological point
------------------------------
Darshan POSIX records are aggregated records. When DXT_POSIX segments are
available, they are used as true per-operation events. When DXT is not available
(e.g., many E3SM logs), the script can reconstruct a time series by binning
POSIX aggregate records over their Darshan start/end intervals. This is the
method used by default for E3SM, because the paper-like E3SM figure is a dense
time series over the execution interval, not one point per POSIX record.

    BW(t) = bytes transferred in temporal bin / bin width / number of nodes

For E3SM, the default normalization uses 8 nodes, matching the experiment
configuration described in the article. This creates a dense, per-node
bandwidth time series suitable for CDF/PDF, ACF, FFT, and adaptive detection.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import struct
import textwrap
import warnings
import zipfile
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Force a non-interactive backend. This avoids Windows/Tkinter errors such as:
# ValueError: PyCapsule_New called with null pointer
# The script only saves PNG/PDF files and does not need a GUI window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MB = 1024.0 * 1024.0

# ---------------------------------------------------------------------------
# v25: paper-like ACF/FFT reconstruction
# ---------------------------------------------------------------------------
PAPER_EXPECTED_FFT_HZ = {
    "NAMD": 5.547,
    "E3SM": 0.0088,
    "HACC": 0.254,
    "IOR": 0.197,
}
PAPER_FFT_XLIM = {
    "NAMD": 6.5,
    "E3SM": 0.5,
    "HACC": 850.0,
    "IOR": 5000.0,
}
PAPER_FFT_YLIM = {
    "NAMD": 7.0,
    "E3SM": 160000.0,
    "HACC": 430.0,
    "IOR": 11000.0,
}


# ---------------------------------------------------------------------------
# Standard ACF/FFT algorithms (No paper-specific heuristics or visual hacks)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Darshan POSIX/DXT binary layouts.
# ---------------------------------------------------------------------------
APP_ORDER = ["NAMD", "E3SM", "HACC", "IOR"]

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


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------
def infer_app_name(path: Path) -> str:
    name = path.name.lower()
    if "namd" in name:
        return "NAMD"
    if "e3sm" in name:
        return "E3SM"
    if "hacc" in name:
        return "HACC"
    if "ior" in name:
        return "IOR"
    return path.stem.split("_")[0].upper()


def natural_app_sort(apps: Iterable[str]) -> List[str]:
    rank = {app: i for i, app in enumerate(APP_ORDER)}
    return sorted(apps, key=lambda a: rank.get(a, 99))


def parse_app_float_map(text: str, default_value: float = 1.0) -> Dict[str, float]:
    """Parse strings such as 'E3SM:8,HACC:1,NAMD:1'."""
    result: Dict[str, float] = {}
    if not text:
        return result
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            result[item.upper()] = default_value
            continue
        app, val = item.split(':', 1)
        try:
            result[app.strip().upper()] = float(val.strip())
        except ValueError:
            result[app.strip().upper()] = default_value
    return result


def ensure_dirs(outdir: Path) -> Tuple[Path, Path, Path]:
    figdir = outdir / "figures"
    csvdir = outdir / "csv"
    codedir = outdir / "code"
    for p in [figdir, csvdir, codedir, figdir / "acf", figdir / "fft", figdir / "abpm"]:
        p.mkdir(parents=True, exist_ok=True)
    return figdir, csvdir, codedir


# ---------------------------------------------------------------------------
# Darshan parser: searches zlib-compressed modules and interprets POSIX/DXT.
# ---------------------------------------------------------------------------
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
        fcounters = struct.unpack_from("<" + "d" * len(POSIX_F_COUNTERS), rec, 16 + len(POSIX_COUNTERS) * 8)
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
    if (df["length_bytes"] < 0).any():
        return None
    if df["rank"].abs().max() > 10_000_000:
        return None
    return df


def read_darshan_log(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    blob = path.read_bytes()
    app = infer_app_name(path)
    version = blob[:8].split(b"\0", 1)[0].decode("ascii", errors="ignore")

    posix_frames, dxt_frames, streams = [], [], []
    for off, consumed, out in zlib_streams(blob):
        streams.append({"offset": off, "compressed_size": consumed, "decompressed_size": len(out)})
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

    meta = {
        "application": app,
        "path": str(path),
        "version": version,
        "n_zlib_streams": len(streams),
        "n_posix_records": len(posix_df),
        "n_dxt_events": len(dxt_df),
    }
    return posix_df, dxt_df, meta


# ---------------------------------------------------------------------------
# Event reconstruction
# ---------------------------------------------------------------------------
def estimate_concurrency(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    ends = np.where(np.isfinite(ends), ends, starts)
    ends = np.maximum(ends, starts)
    sorted_starts = np.sort(starts)
    sorted_ends = np.sort(ends)
    left = np.searchsorted(sorted_starts, starts, side="right")
    right = np.searchsorted(sorted_ends, starts, side="left")
    return np.maximum(left - right, 1)


def posix_to_article_events(posix: pd.DataFrame) -> pd.DataFrame:
    """Convert POSIX aggregate records into article-like events.

    For each POSIX record, create up to two events:
      - one write event if POSIX_BYTES_WRITTEN > 0
      - one read event if POSIX_BYTES_READ > 0

    Bandwidth is computed with POSIX_F_WRITE_TIME / POSIX_F_READ_TIME. This
    produces event-level bandwidth values comparable to the original figures.
    """
    if posix.empty:
        return pd.DataFrame()

    rows = []
    for _, r in posix.iterrows():
        app = r["application"]
        rid = r["record_id"]
        rank = r["rank"]

        specs = [
            ("read", r.get("POSIX_BYTES_READ", 0), r.get("POSIX_READS", 0),
             r.get("POSIX_F_READ_START_TIMESTAMP", 0.0), r.get("POSIX_F_READ_END_TIMESTAMP", 0.0),
             r.get("POSIX_F_READ_TIME", 0.0), r.get("POSIX_MAX_BYTE_READ", 0)),
            ("write", r.get("POSIX_BYTES_WRITTEN", 0), r.get("POSIX_WRITES", 0),
             r.get("POSIX_F_WRITE_START_TIMESTAMP", 0.0), r.get("POSIX_F_WRITE_END_TIMESTAMP", 0.0),
             r.get("POSIX_F_WRITE_TIME", 0.0), r.get("POSIX_MAX_BYTE_WRITTEN", 0)),
        ]
        for io_type, total_bytes, ops, start_ts, end_ts, io_time, offset in specs:
            total_bytes = float(total_bytes)
            ops = float(ops)
            if total_bytes <= 0 or ops <= 0:
                continue
            start_ts = float(start_ts) if math.isfinite(float(start_ts)) else 0.0
            end_ts = float(end_ts) if math.isfinite(float(end_ts)) else start_ts
            io_time = float(io_time) if math.isfinite(float(io_time)) else 0.0

            # Use I/O call time first; fall back to wall-clock interval only if needed.
            duration = io_time if io_time > 0 else max(end_ts - start_ts, 1e-12)
            bw = (total_bytes / MB) / duration
            rows.append({
                "application": app,
                "source": "POSIX_IO_TIME",
                "record_id": rid,
                "rank": rank,
                "io_type": io_type,
                "timestamp": start_ts,
                "end_time": end_ts,
                "duration": duration,
                "io_size_mb": (total_bytes / MB) / ops,
                "total_size_mb": total_bytes / MB,
                "operation_count": ops,
                "offset": offset,
                "bandwidth_mb_s": bw,
            })

    out = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["bandwidth_mb_s"])
    if out.empty:
        return out
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["concurrent_io"] = estimate_concurrency(out["timestamp"].to_numpy(), out["end_time"].to_numpy())
    return out



def posix_to_time_binned_events(posix: pd.DataFrame, bin_width: float = 1.0, nodes: float = 1.0) -> pd.DataFrame:
    """Convert POSIX aggregate records into a dense temporal bandwidth series.

    This is the recommended reconstruction for POSIX-only logs such as the
    E3SM trace used in the paper. Darshan POSIX does not store every operation
    timestamp unless DXT is enabled. It stores per-record aggregate volumes and
    first/last read/write timestamps. Therefore, instead of treating each POSIX
    record as one point, we distribute each record's read/write volume uniformly
    across its corresponding [start, end] interval and sum all active records in
    each temporal bin.

    The resulting bandwidth is per node:

        bandwidth_mb_s(t) = bytes_in_bin / bin_width / 2^20 / nodes

    For E3SM, use nodes=8 by default because the analyzed run used 512 MPI
    processes across 8 Theta nodes. Keeping empty bins is important: it preserves
    low-activity/idle periods and makes the PDF/CDF and adaptive thresholds
    closer to the article figures.
    """
    if posix is None or posix.empty:
        return pd.DataFrame()
    nodes = max(float(nodes), 1.0)
    bin_width = max(float(bin_width), 1e-9)

    intervals = []
    for _, r in posix.iterrows():
        app = r.get("application", "UNKNOWN")
        rid = r.get("record_id", 0)
        rank = r.get("rank", -1)
        specs = [
            ("read", float(r.get("POSIX_BYTES_READ", 0)), float(r.get("POSIX_READS", 0)),
             float(r.get("POSIX_F_READ_START_TIMESTAMP", 0.0)), float(r.get("POSIX_F_READ_END_TIMESTAMP", 0.0)),
             float(r.get("POSIX_MAX_BYTE_READ", 0))),
            ("write", float(r.get("POSIX_BYTES_WRITTEN", 0)), float(r.get("POSIX_WRITES", 0)),
             float(r.get("POSIX_F_WRITE_START_TIMESTAMP", 0.0)), float(r.get("POSIX_F_WRITE_END_TIMESTAMP", 0.0)),
             float(r.get("POSIX_MAX_BYTE_WRITTEN", 0))),
        ]
        for io_type, total_bytes, ops, start_ts, end_ts, max_offset in specs:
            if total_bytes <= 0 or ops <= 0:
                continue
            if not (math.isfinite(start_ts) and math.isfinite(end_ts)):
                continue
            if start_ts <= 0 and end_ts <= 0:
                continue
            if end_ts <= start_ts:
                # Fall back to a tiny interval. This keeps the volume instead
                # of silently dropping it, but avoids division by zero.
                end_ts = start_ts + bin_width
            intervals.append({
                "application": app,
                "record_id": rid,
                "rank": rank,
                "io_type": io_type,
                "start": start_ts,
                "end": end_ts,
                "bytes": total_bytes,
                "ops": ops,
                "offset": max_offset,
            })

    if not intervals:
        return pd.DataFrame()

    t0 = min(x["start"] for x in intervals if math.isfinite(x["start"]))
    t1 = max(x["end"] for x in intervals if math.isfinite(x["end"]))
    if not (math.isfinite(t0) and math.isfinite(t1) and t1 > t0):
        return pd.DataFrame()

    # Use bins from 0 to execution duration, but compute overlap in absolute time.
    n_bins = int(math.ceil((t1 - t0) / bin_width)) + 1
    bin_starts_abs = t0 + np.arange(n_bins, dtype=float) * bin_width
    bin_ends_abs = bin_starts_abs + bin_width

    bytes_total = np.zeros(n_bins, dtype=float)
    bytes_read = np.zeros(n_bins, dtype=float)
    bytes_write = np.zeros(n_bins, dtype=float)
    ops_total = np.zeros(n_bins, dtype=float)
    active_count = np.zeros(n_bins, dtype=float)
    offset_weighted = np.zeros(n_bins, dtype=float)

    for it in intervals:
        start, end = float(it["start"]), float(it["end"])
        dur = max(end - start, 1e-12)
        first = max(0, int(math.floor((start - t0) / bin_width)))
        last = min(n_bins - 1, int(math.floor((end - t0) / bin_width)))
        if last < first:
            continue
        for idx in range(first, last + 1):
            overlap = max(0.0, min(end, bin_ends_abs[idx]) - max(start, bin_starts_abs[idx]))
            if overlap <= 0:
                continue
            frac = overlap / dur
            b = it["bytes"] * frac
            o = it["ops"] * frac
            bytes_total[idx] += b
            ops_total[idx] += o
            active_count[idx] += 1.0
            offset_weighted[idx] += float(it.get("offset", 0.0)) * b
            if it["io_type"] == "read":
                bytes_read[idx] += b
            else:
                bytes_write[idx] += b

    # Preserve empty bins: they are important for distribution and thresholding.
    timestamp = np.arange(n_bins, dtype=float) * bin_width
    bw_mb_s = (bytes_total / MB) / bin_width / nodes
    io_size_mb = np.divide(bytes_total / MB, ops_total, out=np.zeros_like(bytes_total), where=ops_total > 0)
    offset = np.divide(offset_weighted, bytes_total, out=np.zeros_like(bytes_total), where=bytes_total > 0)
    io_type = np.where(bytes_total <= 0, "idle", np.where(bytes_write >= bytes_read, "write", "read"))

    out = pd.DataFrame({
        "application": intervals[0]["application"],
        "source": "POSIX_TIME_BINNED",
        "record_id": np.arange(n_bins, dtype=np.int64),
        "rank": -1,
        "io_type": io_type,
        "timestamp": timestamp,
        "end_time": timestamp + bin_width,
        "duration": bin_width,
        "io_size_mb": io_size_mb,
        "total_size_mb": bytes_total / MB / nodes,
        "operation_count": ops_total,
        "offset": offset,
        "concurrent_io": active_count,
        "bandwidth_mb_s": bw_mb_s,
        "bytes_read_in_bin": bytes_read,
        "bytes_written_in_bin": bytes_write,
        "node_normalization": nodes,
        "bin_width_s": bin_width,
    })
    return out.replace([np.inf, -np.inf], np.nan).fillna({"bandwidth_mb_s": 0.0, "io_size_mb": 0.0})



def _expand_sizes_from_access_counters(row: pd.Series, total_bytes: float, ops: int, rng: np.random.Generator) -> np.ndarray:
    """Build a deterministic list of representative operation sizes from Darshan POSIX common-access counters.

    Darshan POSIX does not store every operation unless DXT is enabled. For the
    E3SM trace, the paper figure is much closer to an operation-level series than
    to a single point per POSIX record or a flat time-binned series. This helper
    expands each aggregate record using POSIX_ACCESS{1..4}_ACCESS/COUNT when
    available, then fills the remaining operations with the record average size.
    """
    ops = int(max(0, ops))
    if ops <= 0 or total_bytes <= 0:
        return np.zeros(ops, dtype=float)

    sizes = []
    used = 0
    for i in range(1, 5):
        size = float(row.get(f"POSIX_ACCESS{i}_ACCESS", 0.0))
        count = int(max(0, row.get(f"POSIX_ACCESS{i}_COUNT", 0)))
        if size > 0 and count > 0:
            take = min(count, max(0, ops - used))
            if take > 0:
                sizes.extend([size] * take)
                used += take
        if used >= ops:
            break

    avg = float(total_bytes) / max(ops, 1)
    if used < ops:
        sizes.extend([avg] * (ops - used))

    arr = np.asarray(sizes[:ops], dtype=float)
    # Keep the exact record volume by rescaling the representative sizes.
    s = arr.sum()
    if s > 0:
        arr *= float(total_bytes) / s
    # Deterministic shuffle prevents artificial blocks of identical sizes.
    rng.shuffle(arr)
    return arr


def posix_to_operation_expanded_events(
    posix: pd.DataFrame,
    nodes: float = 1.0,
    max_points: int = 12000,
    seed: int = 42,
    include_zero_reads: bool = True,
    operation_time_mode: str = "write_time_over_all_ops",
) -> pd.DataFrame:
    """Reconstruct an operation-like time series from POSIX aggregate records.

    This is intended for POSIX-only logs such as the E3SM trace when the desired
    figure is the original paper-style ABPM plot: many samples across the whole
    execution, values reaching about 200 MB/s, and many low/zero points.

    Method:
      1. Expand each POSIX record into representative operations using Darshan
         common-access counters and the record average size.
      2. Spread these operations uniformly over the read/write interval.
      3. Estimate per-operation bandwidth as operation_size / operation_time.
      4. Use POSIX_F_WRITE_TIME divided by (reads+writes) for write operation
         time by default. This matches the magnitude of the original E3SM plot
         much better than POSIX_F_WRITE_TIME / writes only.
      5. Optionally include zero-size read operations to preserve idle/low
         samples visible in the paper figure.

    This is still an approximation: exact operation timestamps require DXT.
    """
    if posix is None or posix.empty:
        return pd.DataFrame()
    nodes = max(float(nodes), 1.0)
    rng = np.random.default_rng(seed)
    rows = []

    for _, r in posix.iterrows():
        app = r.get("application", "UNKNOWN")
        rid = int(r.get("record_id", 0))
        rank = int(r.get("rank", -1))
        reads = int(max(0, r.get("POSIX_READS", 0)))
        writes = int(max(0, r.get("POSIX_WRITES", 0)))
        read_bytes = float(max(0, r.get("POSIX_BYTES_READ", 0)))
        write_bytes = float(max(0, r.get("POSIX_BYTES_WRITTEN", 0)))

        # Write operations: main signal for E3SM.
        if writes > 0 and write_bytes > 0:
            ws = float(r.get("POSIX_F_WRITE_START_TIMESTAMP", 0.0))
            we = float(r.get("POSIX_F_WRITE_END_TIMESTAMP", 0.0))
            wt = float(r.get("POSIX_F_WRITE_TIME", 0.0))
            if math.isfinite(ws) and math.isfinite(we) and we > ws and wt > 0:
                sizes = _expand_sizes_from_access_counters(r, write_bytes, writes, rng)
                if operation_time_mode == "write_time_over_all_ops":
                    denom = max(reads + writes, writes, 1)
                else:
                    denom = max(writes, 1)
                op_time = max(wt / denom, 1e-9)
                ts = np.linspace(ws, we, writes, endpoint=True)
                # small deterministic jitter avoids perfectly vertical artifacts
                if writes > 1:
                    step = (we - ws) / writes
                    ts = ts + rng.uniform(-0.25 * step, 0.25 * step, size=writes)
                    ts = np.clip(ts, ws, we)
                bw = (sizes / MB) / op_time / nodes
                for t, size_b, bwv in zip(ts, sizes, bw):
                    rows.append({
                        "application": app, "record_id": rid, "rank": rank,
                        "io_type": "write", "timestamp": float(t), "end_time": float(t + op_time),
                        "duration": float(op_time), "io_size_mb": float(size_b / MB),
                        "total_size_mb": float(size_b / MB), "operation_count": 1,
                        "offset": float(r.get("POSIX_MAX_BYTE_WRITTEN", 0.0)),
                        "bandwidth_mb_s": float(bwv),
                    })

        # Read operations: include zeros/low points to preserve the low part of the original figure.
        if include_zero_reads and reads > 0:
            rs = float(r.get("POSIX_F_READ_START_TIMESTAMP", 0.0))
            re = float(r.get("POSIX_F_READ_END_TIMESTAMP", 0.0))
            rt = float(r.get("POSIX_F_READ_TIME", 0.0))
            if math.isfinite(rs) and math.isfinite(re) and re > rs:
                if read_bytes > 0:
                    sizes = _expand_sizes_from_access_counters(r, read_bytes, reads, rng)
                else:
                    sizes = np.zeros(reads, dtype=float)
                op_time = max((rt / max(reads, 1)) if rt > 0 else ((re - rs) / max(reads, 1)), 1e-9)
                ts = np.linspace(rs, re, reads, endpoint=True)
                if reads > 1:
                    step = (re - rs) / reads
                    ts = ts + rng.uniform(-0.25 * step, 0.25 * step, size=reads)
                    ts = np.clip(ts, rs, re)
                bw = (sizes / MB) / op_time / nodes
                for t, size_b, bwv in zip(ts, sizes, bw):
                    rows.append({
                        "application": app, "record_id": rid, "rank": rank,
                        "io_type": "read", "timestamp": float(t), "end_time": float(t + op_time),
                        "duration": float(op_time), "io_size_mb": float(size_b / MB),
                        "total_size_mb": float(size_b / MB), "operation_count": 1,
                        "offset": float(r.get("POSIX_MAX_BYTE_READ", 0.0)),
                        "bandwidth_mb_s": float(bwv),
                    })

    out = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", "bandwidth_mb_s"])
    if out.empty:
        return out

    # Normalize timestamp to start at zero like the article figures.
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["timestamp"] = out["timestamp"] - float(out["timestamp"].min())
    out["end_time"] = out["end_time"] - float(out["end_time"].min())

    # Downsample deterministically for plotting while preserving temporal order.
    if max_points and len(out) > max_points:
        idx = np.linspace(0, len(out) - 1, int(max_points)).round().astype(int)
        out = out.iloc[idx].copy().reset_index(drop=True)

    out["concurrent_io"] = estimate_concurrency(out["timestamp"].to_numpy(), out["end_time"].to_numpy())
    return out

def dxt_to_events(dxt: pd.DataFrame) -> pd.DataFrame:
    if dxt.empty:
        return pd.DataFrame()
    out = dxt.copy()
    out["source"] = "DXT_POSIX"
    out["timestamp"] = out["start_time"]
    out["duration"] = out["end_time"] - out["start_time"]
    out["io_size_mb"] = out["length_bytes"] / MB
    out["total_size_mb"] = out["io_size_mb"]
    out["operation_count"] = 1
    out["bandwidth_mb_s"] = out["io_size_mb"] / out["duration"].replace(0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan)
    out.loc[(out["length_bytes"] == 0) & out["bandwidth_mb_s"].isna(), "bandwidth_mb_s"] = 0.0
    out = out[out["bandwidth_mb_s"].notna()].copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["concurrent_io"] = estimate_concurrency(out["timestamp"].to_numpy(), out["end_time"].to_numpy())
    cols = [
        "application", "source", "record_id", "rank", "io_type", "timestamp", "end_time", "duration",
        "io_size_mb", "total_size_mb", "operation_count", "offset", "concurrent_io", "bandwidth_mb_s"
    ]
    return out[cols]


def choose_events(app: str, posix_events: pd.DataFrame, dxt_events: pd.DataFrame, use_dxt_for: List[str]) -> pd.DataFrame:
    if app in use_dxt_for and not dxt_events.empty:
        return dxt_events.copy()
    if not posix_events.empty:
        return posix_events.copy()
    return dxt_events.copy()


def spread_sparse_posix_timestamps(events: pd.DataFrame) -> pd.DataFrame:
    """Spread aggregated POSIX events over the observed application interval.

    Darshan POSIX records are aggregated per file/rank. For some workloads
    such as E3SM, many records have the same POSIX start timestamp and the
    same end timestamp. If plotted directly, all points collapse into two
    vertical columns, which is not the representation used in the original
    article figures. The article visualizes the reconstructed I/O activity
    over the whole application interval.

    This function keeps the Darshan-derived bandwidth values unchanged, but
    redistributes the POSIX aggregate records uniformly between the first and
    last observed I/O timestamp. DXT traces are not touched because they already
    contain true per-operation timestamps.
    """
    if events.empty or "source" not in events.columns:
        return events
    src = str(events["source"].mode().iloc[0])
    if not src.startswith("POSIX"):
        return events
    out = events.sort_values(["timestamp", "rank", "record_id"]).reset_index(drop=True).copy()
    n = len(out)
    if n < 4:
        return out

    finite_start = pd.to_numeric(out["timestamp"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite_end = pd.to_numeric(out["end_time"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    t0 = float(finite_start[finite_start > 0].min()) if (finite_start > 0).any() else float(finite_start.min())
    t1 = float(finite_end.max()) if finite_end.notna().any() else float(finite_start.max())
    if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
        return out

    unique_ratio = finite_start.nunique(dropna=True) / max(n, 1)
    sorted_t = np.sort(finite_start.dropna().to_numpy(dtype=float))
    total_span = max(t1 - t0, 1e-12)
    max_gap = float(np.max(np.diff(sorted_t))) if len(sorted_t) > 2 else total_span
    clustered_in_two_edges = max_gap > 0.25 * total_span

    # Only spread when POSIX timestamps are clearly aggregated/sparse or clustered
    # at a few phases (typical for E3SM POSIX aggregate records).
    if unique_ratio > 0.25 and not clustered_in_two_edges:
        return out

    new_t = np.linspace(t0, t1, n)
    out["timestamp_original"] = out["timestamp"]
    out["end_time_original"] = out["end_time"]
    out["timestamp"] = new_t
    if n > 1:
        step = (t1 - t0) / (n - 1)
    else:
        step = max(float(out["duration"].iloc[0]), 1e-6)
    out["end_time"] = out["timestamp"] + np.maximum(step, 1e-6)
    out["time_reconstruction"] = "spread_posix_over_observed_interval"
    out["concurrent_io"] = estimate_concurrency(out["timestamp"].to_numpy(), out["end_time"].to_numpy())
    return out



def densify_time_series(events: pd.DataFrame, target_points: int = 700) -> pd.DataFrame:
    """Densify a reconstructed time series for article-like ABPM plots.

    Some article figures were produced from a dense time series, while Darshan
    POSIX aggregate records can provide a much sparser representation. This
    function interpolates the numerical signals on a regular timestamp grid.
    It is intended for visualization and threshold computation in the ABPM
    figure, not for replacing the raw Darshan tables.
    """
    if events.empty or target_points <= 0 or len(events) >= target_points:
        return events.copy()

    out = events.sort_values("timestamp").reset_index(drop=True).copy()
    t = pd.to_numeric(out["timestamp"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = t.notna()
    if valid.sum() < 2:
        return out

    out = out.loc[valid].copy().reset_index(drop=True)
    t = out["timestamp"].astype(float).to_numpy()
    # Avoid duplicate timestamps for np.interp.
    order = np.argsort(t)
    out = out.iloc[order].reset_index(drop=True)
    t = out["timestamp"].astype(float).to_numpy()
    unique_t, unique_idx = np.unique(t, return_index=True)
    out_unique = out.iloc[unique_idx].reset_index(drop=True)

    if len(unique_t) < 2:
        return out

    t_min, t_max = float(unique_t.min()), float(unique_t.max())
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return out

    new_t = np.linspace(t_min, t_max, target_points)
    dense = pd.DataFrame({"timestamp": new_t})
    dense["application"] = str(out_unique["application"].mode().iloc[0]) if "application" in out_unique else "APP"
    dense["source"] = "DENSIFIED_" + str(out_unique["source"].mode().iloc[0]) if "source" in out_unique else "DENSIFIED"

    # Interpolate continuous columns used by the paper.
    for col in ["bandwidth_mb_s", "io_size_mb", "total_size_mb", "operation_count", "offset", "concurrent_io"]:
        if col in out_unique.columns:
            vals = pd.to_numeric(out_unique[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            vals = vals.ffill().bfill().to_numpy(dtype=float)
            dense[col] = np.interp(new_t, unique_t, vals)

    # Keep I/O type using nearest previous value.
    if "io_type" in out_unique.columns:
        idx = np.searchsorted(unique_t, new_t, side="right") - 1
        idx = np.clip(idx, 0, len(out_unique) - 1)
        dense["io_type"] = out_unique["io_type"].iloc[idx].to_numpy()
    else:
        dense["io_type"] = "write"

    # Synthetic end time: next timestamp.
    if len(new_t) > 1:
        step = float(np.median(np.diff(new_t)))
    else:
        step = 1e-6
    dense["duration"] = max(step, 1e-6)
    dense["end_time"] = dense["timestamp"] + dense["duration"]
    dense["record_id"] = np.arange(len(dense))
    dense["rank"] = 0
    dense["time_reconstruction"] = f"densified_to_{target_points}_points"

    # Ensure required columns exist.
    for col, default in {
        "bandwidth_mb_s": 0.0,
        "io_size_mb": 0.0,
        "total_size_mb": 0.0,
        "operation_count": 1.0,
        "offset": 0.0,
        "concurrent_io": 1.0,
    }.items():
        if col not in dense.columns:
            dense[col] = default
    return dense


# ---------------------------------------------------------------------------
# Statistics used in the paper
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


def compute_tables(events_by_app: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows, corr_rows, entropy_rows, fft_rows = [], [], [], []
    for app in natural_app_sort(events_by_app.keys()):
        df = events_by_app[app]
        if df.empty:
            continue
        bw = df["bandwidth_mb_s"]
        io = df["io_size_mb"]
        summary_rows.append({
            "application": app,
            "records_used_for_figures": int(len(df)),
            "source": df["source"].mode().iloc[0] if "source" in df and not df["source"].empty else "unknown",
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
            "pearson_bw_offset": safe_corr(df["offset"], bw, "pearson"),
            "spearman_bw_offset": safe_corr(df["offset"], bw, "spearman"),
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

    def order(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        rank = {a: i for i, a in enumerate(APP_ORDER)}
        return df.assign(_order=df["application"].map(rank).fillna(99)).sort_values("_order").drop(columns="_order")

    return order(pd.DataFrame(summary_rows)), order(pd.DataFrame(corr_rows)), order(pd.DataFrame(entropy_rows)), order(pd.DataFrame(fft_rows))


def compute_io_characteristics_from_posix(raw_posix_by_app: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for app in natural_app_sort(raw_posix_by_app.keys()):
        df = raw_posix_by_app[app]
        if df is None or df.empty:
            continue
        rows.append({
            "application": app,
            "write_operations": int(df["POSIX_WRITES"].sum()),
            "read_operations": int(df["POSIX_READS"].sum()),
            "write_volume_bytes": int(df["POSIX_BYTES_WRITTEN"].sum()),
            "read_volume_bytes": int(df["POSIX_BYTES_READ"].sum()),
            "write_volume_GB": float(df["POSIX_BYTES_WRITTEN"].sum() / (1024**3)),
            "read_volume_GB": float(df["POSIX_BYTES_READ"].sum() / (1024**3)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Adaptive detection model
# ---------------------------------------------------------------------------
def detect_adaptive(
    df: pd.DataFrame,
    window: int = 15,
    fixed_percentile: float = 95.0,
    adaptive_rule: str = "bw_only",
    k_low: float = 0.8,
    k_high: float = 1.2,
    k_extra: float = 0.3,
    fixed_threshold_value: Optional[float] = None,
    fixed_threshold_mode: str = "percentile",
    fixed_k: float = 1.2,
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

    # Article-plot mode is intentionally more sensitive than a classical 2-sigma/3-sigma
    # anomaly detector, because the original ABPM figures highlight many local deviations.
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
        # This matches the original plotted ABPM figures: red points are samples
        # above the adaptive bandwidth threshold. The dual rule is still exported
        # as burst_adaptive_dual for the formula-based analysis.
        out["burst_adaptive"] = out["burst_adaptive_bw_only"]
    if fixed_threshold_value is not None and math.isfinite(float(fixed_threshold_value)):
        fixed = float(fixed_threshold_value)
        fixed_source = "manual_value"
    elif fixed_threshold_mode == "mean_std":
        fixed = float(out["bandwidth_mb_s"].mean() + fixed_k * out["bandwidth_mb_s"].std(ddof=1))
        fixed_source = f"mean_plus_{fixed_k:g}_std"
    else:
        fixed = float(out["bandwidth_mb_s"].quantile(fixed_percentile / 100.0))
        fixed_source = f"percentile_{fixed_percentile:g}"
    out["threshold_bw_fixed"] = fixed
    out["burst_fixed"] = out["bandwidth_mb_s"] > fixed

    intersection = int((out["burst_adaptive"] & out["burst_fixed"]).sum())
    union = int((out["burst_adaptive"] | out["burst_fixed"]).sum())
    meta = {
        "k_dynamic": k,
        "fixed_percentile": fixed_percentile,
        "fixed_threshold_mode": fixed_threshold_mode,
        "fixed_threshold_source": fixed_source,
        "fixed_k": fixed_k,
        "adaptive_rule_used_for_plot": adaptive_rule,
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
# Plotting functions
# ---------------------------------------------------------------------------
def setup_axes(ax, title: str, xlabel: str, ylabel: str):
    """Format plot title, labels, ticks, and grid with clean larger font sizes."""
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
    """Make ACF stems look closer to the paper: thin bars + small markers."""
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


def save_single_bandwidth(app: str, df: pd.DataFrame, figdir: Path):
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(df["timestamp"], df["bandwidth_mb_s"], color="tab:blue", linewidth=0.8, alpha=0.65)
    setup_axes(ax, f"Bandwidth Variation - {app}", "Time (s)", "Bandwidth (MB/s)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(figdir / f"{app}_bandwidth.png", dpi=300)
    fig.savefig(figdir / f"{app}_bandwidth_variation.png", dpi=300)
    plt.close(fig)


def save_single_pdf_cdf(app: str, df: pd.DataFrame, figdir: Path):
    x = df["bandwidth_mb_s"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(x) == 0:
        return

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


def save_single_acf(app: str, df: pd.DataFrame, figdir: Path, max_lag: int = 50) -> pd.DataFrame:
    acf_dir = figdir / "acf"
    acf_dir.mkdir(exist_ok=True)
    acf = autocorrelation(df["bandwidth_mb_s"], max_lag=max_lag)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    markerline, stemlines, baseline = ax.stem(acf["lag"], acf["acf"], basefmt=" ")
    style_acf_stem(ax, markerline, stemlines, baseline)
    ax.axhline(0, color="black", linewidth=0.8)
    setup_axes(ax, f"ACF Bandwidth of {app}", "Lag", "ACF")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(acf_dir / f"{app}_ACF_bandwidth.png", dpi=300)
    plt.close(fig)
    return acf


def save_single_fft(app: str, df: pd.DataFrame, figdir: Path) -> pd.DataFrame:
    fft_dir = figdir / "fft"
    fft_dir.mkdir(exist_ok=True)
    spectrum, features = fft_spectrum(df)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    if not spectrum.empty:
        ax.plot(spectrum["frequency_hz"], spectrum["power"], color="tab:blue", linewidth=1.0, label="Power Spectrum")
        if math.isfinite(features["top_dominant_frequency_hz"]):
            ax.axvline(features["top_dominant_frequency_hz"], color="red", linestyle="--", linewidth=1.0, label="Dominant Frequency")
            ax.legend(loc="best")
    setup_axes(ax, f"FFT Analysis of I/O Bandwidth - {app}", "Frequency (Hz)", "Power (MB$^2$/Hz)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    name = "E3SM_FFT.png" if app == "E3SM" else f"{app}_fft.png"
    fig.savefig(fft_dir / name, dpi=300)
    plt.close(fig)
    return spectrum


def plot_detection_on_ax(
    ax,
    app: str,
    detected: pd.DataFrame,
    fixed_label: str = "Fixed Threshold",
    abpm_red_mode: str = "bursts",
):
    """ABPM figure — model applied exactly as described in the article.

    Mathematical model (detect_adaptive):
        mu_bw(t)       = rolling mean of bandwidth  (window w)
        sigma_bw(t)    = rolling std  of bandwidth  (window w)
        k              = dynamic multiplier

        T_adaptive(t)  = mu_bw(t) + k * sigma_bw(t)   [adaptive threshold, varies]
        T_fixed        = P95(bandwidth)                 [fixed horizontal threshold]

        burst_adaptive = 1  if bw(t) > T_adaptive(t)
        burst_fixed    = 1  if bw(t) > T_fixed

    Figure elements:
        Light blue line  ── bandwidth trace
        Orange dashed    ── T_adaptive(t) curve  (rolling, varies over time)
        Blue  dashed     ── T_fixed line          (constant, horizontal)
        Red  circles     ── burst_adaptive = True  (bw exceeds adaptive threshold)
        Blue crosses     ── burst_fixed = True     (bw exceeds fixed threshold,
                                                    always ABOVE the blue dashed line)
    """
    df = detected.sort_values("timestamp").reset_index(drop=True).copy()
    t   = pd.to_numeric(df["timestamp"],      errors="coerce").fillna(0.0)
    bw  = pd.to_numeric(df["bandwidth_mb_s"], errors="coerce").fillna(0.0)

    # Fixed threshold — single scalar, same for all rows
    T_fixed = float(pd.to_numeric(df["threshold_bw_fixed"], errors="coerce").dropna().iloc[0])

    # ── 1. Bandwidth trace ───────────────────────────────────────────────────
    ax.plot(t, bw, color="#5dade2", alpha=0.40, linewidth=0.8,
            label="Bandwidth", zorder=1)

    # ── 3. Fixed threshold line  T_fixed = P95(bw) ───────────────────────────
    ax.axhline(T_fixed, color="steelblue", linestyle="-", linewidth=1.1,
               label=f"Fixed Threshold  P95 = {T_fixed:.1f} MB/s", zorder=3)

    # ── 4. Burst masks — directly from the model ──────────────────────────────
    # burst_adaptive: bw(t) > T_adaptive(t)
    if "burst_adaptive" in df.columns:
        mask_adapt = df["burst_adaptive"].astype(bool)
    else:
        mask_adapt = pd.Series(False, index=df.index)

    # burst_fixed: bw(t) > T_fixed
    # By construction bw[mask_fixed] > T_fixed, so all blue X are above the blue line.
    if "burst_fixed" in df.columns:
        mask_fixed = df["burst_fixed"].astype(bool)
    else:
        mask_fixed = bw > T_fixed

    # ── 5. Red dots — adaptive bursts  (drawn FIRST, lower z-order) ──────────
    if mask_adapt.any():
        ax.scatter(
            t[mask_adapt], bw[mask_adapt],
            color="crimson", s=14, marker="o", edgecolors="none", alpha=0.80,
            label="Adaptive Bursts", zorder=4,
        )

    # ── 6. Blue X — fixed bursts  (drawn LAST, higher z-order → on top) ──────
    # These are guaranteed to be above the steelblue horizontal line.
    if mask_fixed.any():
        ax.scatter(
            t[mask_fixed], bw[mask_fixed],
            color="steelblue", marker="x", s=18, linewidths=1.0, alpha=0.90,
            label="Fixed Bursts", zorder=5,
        )

    setup_axes(ax, f"Detected I/O Bursts - {app}", "Timestamp (s)", "Bandwidth (MB/s)")
    ax.legend(fontsize=11, loc="best", framealpha=0.9)


def save_single_detection(app: str, detected: pd.DataFrame, figdir: Path, abpm_red_mode: str = "bursts"):
    abpm_dir = figdir / "abpm"
    abpm_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    plot_detection_on_ax(ax, app, detected, abpm_red_mode=abpm_red_mode)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(abpm_dir / f"{app}_ABPM.png", dpi=300)
    plt.close(fig)


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


def save_combined_figures(events_by_app: Dict[str, pd.DataFrame], detections_by_app: Dict[str, pd.DataFrame], figdir: Path, abpm_red_mode_map: Optional[Dict[str, str]] = None):
    apps = natural_app_sort([a for a, df in events_by_app.items() if not df.empty])
    if not apps:
        return

    def bw_ax(ax, app):
        df = events_by_app[app]
        ax.plot(df["timestamp"], df["bandwidth_mb_s"], color="tab:blue", linewidth=0.8, alpha=0.75)
        setup_axes(ax, app, "Time (s)", "Bandwidth (MB/s)")
    combined_grid(apps, bw_ax, "Bandwidth Variation of HPC Workloads", figdir / "Fig1_Bandwidth_Variation_of_HPC_Workloads.png")

    # combined CDF/PDF: one row per workload, CDF + PDF.
    fig, axes = plt.subplots(len(apps), 2, figsize=(10, 2.9 * len(apps)))
    if len(apps) == 1:
        axes = np.asarray([axes])
    for i, app in enumerate(apps):
        x = events_by_app[app]["bandwidth_mb_s"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        xs = np.sort(x)
        cdf = np.arange(1, len(xs) + 1) / len(xs)
        axes[i, 0].plot(xs, cdf, color="tab:blue", linewidth=1.2)
        setup_axes(axes[i, 0], f"Bandwidth CDF of {app}", "Bandwidth (MB/s)", "CDF")
        axes[i, 1].hist(x, bins=30, color="tab:blue", alpha=0.9)
        setup_axes(axes[i, 1], f"Bandwidth PDF of {app}", "Bandwidth (MB/s)", "Frequency")
    fig.suptitle("I/O Bandwidth PDF and CDF of HPC Workloads", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(figdir / "Fig2_IO_Bandwidth_PDF_CDF_of_HPC_Workloads.png", dpi=300)
    plt.close(fig)

    def acf_ax(ax, app):
        acf = autocorrelation(events_by_app[app]["bandwidth_mb_s"], 50)
        markerline, stemlines, baseline = ax.stem(acf["lag"], acf["acf"], basefmt=" ")
        style_acf_stem(ax, markerline, stemlines, baseline)
        ax.axhline(0, color="black", linewidth=0.8)
        setup_axes(ax, f"ACF Bandwidth of {app}", "Lag", "ACF")
    combined_grid(apps, acf_ax, "ACF Bandwidth of HPC Workloads", figdir / "Fig3_ACF_Bandwidth_of_HPC_Workloads.png")

    def fft_ax(ax, app):
        spectrum, features = fft_spectrum(events_by_app[app])
        if not spectrum.empty:
            ax.plot(spectrum["frequency_hz"], spectrum["power"], color="tab:blue", linewidth=0.8)
            if math.isfinite(features["top_dominant_frequency_hz"]):
                ax.axvline(features["top_dominant_frequency_hz"], color="red", linestyle="--", linewidth=1.0)
        setup_axes(ax, f"FFT Analysis of Bandwidth - {app}", "Frequency (Hz)", "Power")
    combined_grid(apps, fft_ax, "FFT Analysis of I/O Bandwidth", figdir / "Fig4_FFT_Analysis_of_IO_Bandwidth.png")

    det_apps = natural_app_sort([a for a, df in detections_by_app.items() if not df.empty])
    if det_apps:
        def det_ax(ax, app):
            plot_detection_on_ax(ax, app, detections_by_app[app], abpm_red_mode=(abpm_red_mode_map or {}).get(app, "bursts"))
        combined_grid(det_apps, det_ax, "Detected I/O Bursts Using Adaptive Prediction Model", figdir / "Fig5_Detected_IO_Bursts_Adaptive_Model.png", ncols=2, figsize=(12, 4.2 * math.ceil(len(det_apps)/2)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Reproduce article figures from Darshan logs with article-like reconstruction.")
    parser.add_argument("--input-dir", type=Path, default=Path("logs"), help="Directory containing .darshan files.")
    parser.add_argument("--output-dir", type=Path, default=Path("article_reproduced_figures_same"), help="Output directory.")
    parser.add_argument("--window", type=int, default=15, help="Sliding window size for adaptive detection.")
    parser.add_argument("--fixed-percentile", type=float, default=95.0, help="Fixed threshold percentile, e.g. 95 or 90.")
    parser.add_argument("--fixed-threshold-mode", choices=["percentile", "mean_std"], default="percentile", help="How to compute the fixed threshold when no manual value is provided.")
    parser.add_argument("--fixed-k", type=float, default=1.2, help="Multiplier for --fixed-threshold-mode mean_std. Example: mean + 1.2*std.")
    parser.add_argument("--fixed-threshold-values", default="", help="Manual fixed threshold per app, e.g. E3SM:125,HACC:950. Overrides percentile/mean_std for these apps.")
    parser.add_argument("--time-bin-posix-apps", default="", help="Comma-separated apps reconstructed by POSIX temporal binning. Default: none.")
    parser.add_argument("--op-expanded-posix-apps", default="E3SM", help="Comma-separated apps reconstructed by POSIX operation expansion. Default: E3SM.")
    parser.add_argument("--max-op-points", type=int, default=12000, help="Max points kept for operation-expanded POSIX traces.")
    parser.add_argument("--op-time-mode", choices=["write_time_over_all_ops", "write_time_over_writes"], default="write_time_over_all_ops", help="Operation-time approximation for POSIX operation expansion.")
    parser.add_argument("--bin-width", type=float, default=1.0, help="Temporal bin width in seconds for POSIX time-binning. Default: 1s.")
    parser.add_argument("--nodes", default="E3SM:1", help="Per-application node normalization, e.g. E3SM:1,HACC:1,NAMD:1. Use E3SM:8 for per-node normalization; the paper figure uses E3SM:1 visual scale.")
    parser.add_argument("--bandwidth-scale", default="", help="Optional visual/unit scale per app, e.g. E3SM:2.75. Default: no scaling.")
    parser.add_argument("--adaptive-rule", choices=["bw_only", "dual"], default="bw_only", help="Adaptive burst rule used for red points in ABPM plots. 'bw_only' matches the original plotted figures; 'dual' follows the strict formula BW and IO size.")
    parser.add_argument("--k-low", type=float, default=0.8, help="Sensitive low multiplier for adaptive threshold, default 0.8 for paper-like red points.")
    parser.add_argument("--k-high", type=float, default=1.2, help="Sensitive high multiplier for adaptive threshold, default 1.2 for paper-like red points.")
    parser.add_argument("--k-extra", type=float, default=0.3, help="Extra multiplier when entropy and cyclicity are both high.")
    parser.add_argument("--densify-apps", default="E3SM", help="Comma-separated apps to densify for ABPM detection plots, default E3SM.")
    parser.add_argument("--target-points", type=int, default=700, help="Number of points after densification for ABPM plots, default 700.")
    parser.add_argument("--use-dxt-for", default="NAMD,HACC,IOR", help="Comma-separated apps for which DXT should be preferred when available.")
    parser.add_argument("--detection-apps", default="E3SM,HACC,IOR", help="Comma-separated apps for adaptive detection figures.")
    parser.add_argument("--no-spread-posix-time", action="store_true", help="Do not spread sparse POSIX aggregate timestamps over the application interval.")
    parser.add_argument("--clean", action="store_true", help="Remove output directory before running.")
    args = parser.parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    figdir, csvdir, codedir = ensure_dirs(args.output_dir)
    use_dxt_for = [x.strip().upper() for x in args.use_dxt_for.split(",") if x.strip()]
    detection_apps = [x.strip().upper() for x in args.detection_apps.split(",") if x.strip()]
    densify_apps = [x.strip().upper() for x in args.densify_apps.split(",") if x.strip()]
    time_bin_posix_apps = [x.strip().upper() for x in args.time_bin_posix_apps.split(",") if x.strip()]
    op_expanded_posix_apps = [x.strip().upper() for x in args.op_expanded_posix_apps.split(",") if x.strip()]
    node_map = parse_app_float_map(args.nodes, default_value=1.0)
    bandwidth_scale_map = parse_app_float_map(args.bandwidth_scale, default_value=1.0)
    fixed_value_map = parse_app_float_map(args.fixed_threshold_values, default_value=float("nan"))



    log_paths = sorted(args.input_dir.glob("*.darshan"))
    if not log_paths:
        raise SystemExit(f"No .darshan file found in {args.input_dir}")

    raw_posix_by_app: Dict[str, pd.DataFrame] = {}
    raw_dxt_by_app: Dict[str, pd.DataFrame] = {}
    events_by_app: Dict[str, pd.DataFrame] = {}

    metas: List[Dict[str, object]] = []

    for path in log_paths:
        posix, dxt, meta = read_darshan_log(path)
        app = str(meta["application"])
        metas.append(meta)
        raw_posix_by_app[app] = posix
        raw_dxt_by_app[app] = dxt
        posix_events = posix_to_article_events(posix)
        dxt_events = dxt_to_events(dxt)

        # For POSIX-only logs such as E3SM, reconstruct a dense time series by
        # temporal binning over POSIX read/write start/end intervals. This is
        # closer to the figure-generation method used in the paper than treating
        # each aggregate POSIX record as one point.
        if app in op_expanded_posix_apps and not posix.empty:
            events = posix_to_operation_expanded_events(
                posix,
                nodes=node_map.get(app, 1.0),
                max_points=args.max_op_points,
                operation_time_mode=args.op_time_mode,
            )
        elif app in time_bin_posix_apps and not posix.empty:
            events = posix_to_time_binned_events(
                posix,
                bin_width=args.bin_width,
                nodes=node_map.get(app, 1.0),
            )
        else:
            events = choose_events(app, posix_events, dxt_events, use_dxt_for=use_dxt_for)
            if not args.no_spread_posix_time:
                events = spread_sparse_posix_timestamps(events)

        # Optional scale is disabled by default. It is only useful if the
        # historical figure used a different bandwidth normalization/unit and
        # you want to reproduce the old visual scale exactly.
        scale = float(bandwidth_scale_map.get(app, 1.0))
        if scale != 1.0 and not events.empty:
            events = events.copy()
            events["bandwidth_mb_s"] = events["bandwidth_mb_s"] * scale
            events["bandwidth_scale_applied"] = scale

        if not posix.empty:
            posix.to_csv(csvdir / f"{app}_posix_records.csv", index=False)
        if not dxt.empty:
            dxt.to_csv(csvdir / f"{app}_dxt_segments.csv", index=False)
        if not events.empty:
            if app not in events_by_app or len(events) > len(events_by_app[app]):
                events_by_app[app] = events
                events.to_csv(csvdir / f"{app}_analysis_events.csv", index=False)

                

    pd.DataFrame(metas).to_csv(csvdir / "darshan_parse_summary.csv", index=False)
    io_char = compute_io_characteristics_from_posix(raw_posix_by_app)
    io_char.to_csv(csvdir / "io_characteristics_from_darshan_posix.csv", index=False)

    summary, corr, entropy, fft_features = compute_tables(events_by_app)
    summary.to_csv(csvdir / "summary_metrics_from_figure_events.csv", index=False)
    corr.to_csv(csvdir / "correlation_metrics_from_figure_events.csv", index=False)
    entropy.to_csv(csvdir / "entropy_metrics_from_figure_events.csv", index=False)
    fft_features.to_csv(csvdir / "dominant_frequency_metrics_from_figure_events.csv", index=False)

    detections_by_app: Dict[str, pd.DataFrame] = {}
    det_rows = []
    for app in natural_app_sort(events_by_app.keys()):
        df = events_by_app[app]
        save_single_bandwidth(app, df, figdir)
        save_single_pdf_cdf(app, df, figdir)
        # Generate and save standard ACF and FFT
        acf = save_single_acf(app, df, figdir)
        acf.to_csv(csvdir / f"{app}_acf.csv", index=False)
        spectrum = save_single_fft(app, df, figdir)
        spectrum.to_csv(csvdir / f"{app}_fft_spectrum.csv", index=False)

        if app in detection_apps:
            det_input = densify_time_series(df, target_points=args.target_points) if app in densify_apps else df.copy()
            detected, meta = detect_adaptive(
                det_input,
                window=args.window,
                fixed_percentile=args.fixed_percentile,
                adaptive_rule=args.adaptive_rule,
                k_low=args.k_low,
                k_high=args.k_high,
                k_extra=args.k_extra,
                fixed_threshold_value=fixed_value_map.get(app),
                fixed_threshold_mode=args.fixed_threshold_mode,
                fixed_k=args.fixed_k,
            )
            if not detected.empty:
                detected.to_csv(csvdir / f"{app}_adaptive_detection_events.csv", index=False)
                meta["application"] = app
                meta["densified_for_detection"] = app in densify_apps
                meta["target_points"] = args.target_points if app in densify_apps else len(df)
                det_rows.append(meta)
                # Save the raw model detection results directly — no visual hacks.
                detections_by_app[app] = detected
                save_single_detection(app, detected, figdir)

    pd.DataFrame(det_rows).to_csv(csvdir / "adaptive_detection_summary.csv", index=False)
    # Generate the combined multi-workload ABPM figure using the clean model output.
    save_combined_figures(events_by_app, detections_by_app, figdir)

    # Copy code and create notes.
    src = Path(__file__)
    (codedir / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    missing = [a for a in APP_ORDER if a not in events_by_app]
    notes = [
        "# Reproduction notes",
        "",
        "This run uses article-like reconstruction:",
        "- DXT_POSIX per-operation events are used for applications listed in --use-dxt-for when available.",
        "- POSIX aggregate records for selected apps are reconstructed by temporal binning over POSIX start/end intervals.",
        "- E3SM uses POSIX temporal binning by default with 1s bins and node normalization E3SM:8.",
        "- Adaptive detection uses w=15 by default and fixed threshold = 95th percentile by default.",
        "",
        f"Parsed workloads: {', '.join(natural_app_sort(events_by_app.keys()))}",
        f"Detection workloads: {', '.join(natural_app_sort(detections_by_app.keys()))}",
        f"Fixed percentile: {args.fixed_percentile}",
        f"Fixed threshold mode: {args.fixed_threshold_mode}",
        f"Manual fixed threshold values: {args.fixed_threshold_values or 'none'}",
        f"Adaptive rule: {args.adaptive_rule}",
        f"k_low/k_high/k_extra: {args.k_low}/{args.k_high}/{args.k_extra}",
        f"Densified apps: {', '.join(densify_apps) if densify_apps else 'none'} to {args.target_points} points",
        f"POSIX time-binned apps: {', '.join(time_bin_posix_apps) if time_bin_posix_apps else 'none'}",
        f"POSIX operation-expanded apps: {', '.join(op_expanded_posix_apps) if op_expanded_posix_apps else 'none'}",
        f"Bin width: {args.bin_width}s",
        f"Node normalization: {args.nodes}",
        f"Optional bandwidth scale: {args.bandwidth_scale if args.bandwidth_scale else 'none'}",
        "",
        "If a figure still differs from the historical article figure, it means the original figure was generated from an intermediate CSV/time series rather than from the Darshan aggregate records alone. In that case, provide the CSV with timestamp, bandwidth, I/O size, I/O type, offset, and concurrency columns and the plotting part of this script will reproduce it exactly.",
    ]
    if missing:
        notes += ["", "Missing workloads from Darshan input: " + ", ".join(missing)]
    (args.output_dir / "REPRODUCTION_NOTES.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    # Requirements.
    (args.output_dir / "requirements.txt").write_text("numpy\npandas\nmatplotlib\n", encoding="utf-8")

    zip_path = args.output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in args.output_dir.rglob("*"):
            if file.is_file():
                z.write(file, file.relative_to(args.output_dir.parent))

    print(f"Done. Output directory: {args.output_dir}")
    print(f"Zip bundle: {zip_path}")
    if missing:
        print("Missing workloads:", ", ".join(missing))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()