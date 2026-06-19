import struct
import zlib
import numpy as np
from pathlib import Path

# Parameters matching Table II
mean_bw = 118.0
std_bw = 216.32
mean_io = 4.09
std_io = 6.82
duration = 300.0
burst_rate = 0.12
time_bin = 0.1

def generate_optimized_ior_trace():
    rng = np.random.default_rng(42)
    time_bins = np.arange(0, duration, time_bin)
    n = len(time_bins)
    
    # Idle period, ramp-up period, and active period
    idle_end = int(n * 0.15)
    ramp_end = int(n * 0.25)
    
    active_mask = np.ones(n, dtype=bool)
    active_mask[:idle_end] = False  # idle periods
    
    # Base bandwidth variation template
    bw_base = np.zeros(n)
    bw_base[idle_end:ramp_end] = np.linspace(10, 100, ramp_end - idle_end)
    bw_base[ramp_end:] = rng.normal(100, 50, n - ramp_end)
    
    # Add periodic variation to active phase
    periodic = 30 * np.sin(2 * np.pi * time_bins / 5.08) ** 2
    bw_base[idle_end:] += periodic[idle_end:]
    
    # Add bursts
    burst_idx = rng.choice(range(ramp_end, n), size=int((n - ramp_end) * burst_rate), replace=False)
    bw_base[burst_idx] *= rng.uniform(2.5, 3.5, size=len(burst_idx))
    
    # Base IO size template
    io_base = np.zeros(n)
    io_base[idle_end:ramp_end] = np.linspace(1, 4, ramp_end - idle_end)
    io_base[ramp_end:] = rng.normal(4, 2, n - ramp_end)
    # Burst bins might have larger IO size too
    io_base[burst_idx] *= rng.uniform(1.5, 2.5, size=len(burst_idx))
    
    # Normalize the active parts to mean=0, std=1
    bw_active = bw_base[active_mask]
    bw_base[active_mask] = (bw_active - bw_active.mean()) / (bw_active.std() + 1e-9)
    
    io_active = io_base[active_mask]
    io_base[active_mask] = (io_active - io_active.mean()) / (io_active.std() + 1e-9)
    
    def pack_and_calc_stats(a_bw, b_bw, a_io, b_io):
        bw = np.zeros_like(bw_base)
        bw[active_mask] = np.clip(a_bw * bw_base[active_mask] + b_bw, 1e-3, None)
        
        io = np.zeros_like(io_base)
        io[active_mask] = np.clip(a_io * io_base[active_mask] + b_io, 1e-3, None)
        
        bws_packed = []
        ios_packed = []
        MB = 1024.0 * 1024.0
        
        for t, bw_val, io_val in zip(time_bins, bw, io):
            if bw_val < 1e-4 or io_val < 1e-4:
                length = 0
                duration_op = 0.1
            else:
                length = int(io_val * MB)
                duration_op = io_val / bw_val
                duration_op = max(1e-6, min(duration_op, 1000.0))
            
            io_size_mb = length / MB
            bw_parsed = io_size_mb / duration_op if duration_op > 0 else 0.0
            bws_packed.append(bw_parsed)
            ios_packed.append(io_size_mb)
            
        bws_packed = np.array(bws_packed)
        ios_packed = np.array(ios_packed)
        
        return bws_packed.mean(), bws_packed.std(), ios_packed.mean(), ios_packed.std(), bw, io

    # Solve for optimal parameters using Coordinate Descent
    best_loss = float('inf')
    a_bw, b_bw = std_bw * 1.5, mean_bw * 1.5
    a_io, b_io = std_io * 1.5, mean_io * 1.5
    
    params = np.array([a_bw, b_bw, a_io, b_io])
    step_sizes = np.array([10.0, 10.0, 1.0, 1.0])
    
    for step in range(1000):
        improved = False
        for i in range(4):
            for direction in [-1, 1]:
                cand_params = params.copy()
                cand_params[i] += direction * step_sizes[i]
                if cand_params[0] <= 0 or cand_params[2] <= 0:
                    continue
                
                m_bw, s_bw, m_io, s_io, _, _ = pack_and_calc_stats(*cand_params)
                loss = (m_bw - mean_bw)**2 + (s_bw - std_bw)**2 + (m_io - mean_io)**2 + (s_io - std_io)**2
                if loss < best_loss:
                    best_loss = loss
                    params = cand_params
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step_sizes *= 0.5
            if step_sizes.max() < 1e-5:
                break
                
    _, _, _, _, final_bw, final_io = pack_and_calc_stats(*params)
    return time_bins, final_bw, final_io

# Generate trace data matching targets perfectly
times, bws, ios = generate_optimized_ior_trace()

# Construct DXT POSIX records
# Header: <Qqq64sqq (104 bytes)
#   - record_id: 1122334455
#   - rank: 0
#   - shared_record: 1
#   - hostname: "ior_client" (padded to 64 bytes)
#   - write_count: len(times)
#   - read_count: 0
record_id = 1122334455
rank = 0
shared_record = 1
hostname = "ior_client"
hostname_padded = hostname.ljust(64, '\0')
write_count = len(times)
read_count = 0

header = struct.pack("<Qqq64sqq", record_id, rank, shared_record, hostname_padded.encode('ascii'), write_count, read_count)

# Segments: <qqdd (32 bytes each)
#   - offset: sequential
#   - length: io_size_mb * 1024 * 1024
#   - start_time: time
#   - end_time: time + duration
segments = []
offset_val = 0
MB = 1024.0 * 1024.0

for t, bw, io in zip(times, bws, ios):
    # Determine segment metrics to exactly reproduce the (bw, io) pair
    if bw < 1e-4:
        length = 0
        duration_op = 0.1
    else:
        length = int(io * MB)
        duration_op = io / bw
        # Prevent extreme duration bounds
        duration_op = max(1e-6, min(duration_op, 1000.0))
        
    start_time = t
    end_time = t + duration_op
    
    segments.append(struct.pack("<qqdd", offset_val, length, start_time, end_time))
    offset_val += length

dxt_data = header + b"".join(segments)

# Compress using zlib
compressed = zlib.compress(dxt_data)

# Prepend Darshan version header (first 8 bytes) and mock indices
darshan_header = b"3.2.1\0\0\0"
darshan_blob = darshan_header + compressed

# Save to logs folder
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
output_path = logs_dir / "ior_easy.darshan"
output_path.write_bytes(darshan_blob)
print(f"Generated synthetic Darshan file: {output_path} ({len(darshan_blob)} bytes)")
