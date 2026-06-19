# I/O Burst Characterisation and Prediction in HPC Systems

This repository contains a Python implementation to parse Darshan logs directly and reproduce the scientific analysis, figures, and metrics described in the article:
> **"Adaptive I/O Burst Characterization and Prediction in High-Performance Computing Systems"**

The project is designed to run standalone with zero external C dependencies, parsing binary Darshan log files directly (POSIX and DXT modules) to characterize workload I/O behavior and predict I/O bursts using an adaptive prediction model.

---

## Key Features

- **Direct Binary Parser**: Custom binary parser for Darshan logs (handling POSIX and DXT records) written in Python without requiring the heavy C-based `darshan-parser` utility or the `darshan` library.
- **Advanced Temporal Reconstruction**:
  - Reconstructs aggregate POSIX records via temporal binning and operation-expansion to recreate high-fidelity dense bandwidth time series.
  - Utilizes per-operation DXT (Darshan eXtended Tracing) logs when available (e.g., for NAMD, HACC, and IOR).
- **Statistical Analysis**: Computes bandwidth CDF/PDF distributions, Autocorrelation Function (ACF) lag profiles, and Fast Fourier Transform (FFT) power spectra.
- **Adaptive Burst Prediction Model (ABPM)**: Implements the adaptive sliding window thresholding algorithm:
  $$T_{\text{adaptive}}(t) = \mu_{\text{bw}}(t) + k \cdot \sigma_{\text{bw}}(t)$$
  and compares it against fixed threshold benchmarks ($P_{95}$).

---

## Installation

The tool requires standard data science libraries (`numpy`, `pandas`, `matplotlib`).

1. Clone this repository:
   ```bash
   git clone https://github.com/hocinemahni/io_burst_characterisation.git
   cd io_burst_characterisation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Place your Darshan logs
Create a folder named `logs/` in the root directory and place your `.darshan` log files there:
```text
logs/
  e3sm_xxxx.darshan
  hacc_xxxx.darshan
  namd_xxxx.darshan
```

### 2. Run the reproduction pipeline
Execute the pipeline script to parse the logs, execute the mathematical models, and generate all tables and figures:
```bash
python reproduce_article_figures.py --input-dir logs --output-dir output_results --clean
```

### Command Line Arguments
- `--input-dir`: Directory containing `.darshan` files (default: `logs`).
- `--output-dir`: Directory where CSV tables and figures will be saved (default: `output_results`).
- `--window`: Sliding window size ($w$) for adaptive detection (default: `15`).
- `--fixed-percentile`: Percentile value for the fixed threshold baseline (default: `95.0`).
- `--adaptive-rule`: Rule for adaptive burst markers (`bw_only` or `dual`).
- `--k-low` / `--k-high` / `--k-extra`: Multipliers for the adaptive threshold envelope.
- `--clean`: Clean the output directory before starting.

---

## Generated Outputs

The pipeline generates the following assets in your output directory:

### 1. Numerical CSV Data (`csv/`)
- `darshan_parse_summary.csv`: Parsing summary and record counts.
- `io_characteristics_from_darshan_posix.csv`: Statistical summaries of POSIX trace records.
- `summary_metrics_from_figure_events.csv`: Metrics including average, max, and variance bandwidths.
- `correlation_metrics_from_figure_events.csv`: Autocorrelation characteristics.
- `entropy_metrics_from_figure_events.csv`: Shannon entropy values for each workload.
- `dominant_frequency_metrics_from_figure_events.csv`: Dominant frequencies extracted via FFT analysis.
- `adaptive_detection_summary.csv`: Summary metrics of detected burst events.

### 2. Scientific Figures (`figures/`)
- **Figure 1 (Bandwidth Variation)**: High-resolution time-series trace of aggregate IO bandwidth for each workload.
- **Figure 2 (PDF & CDF)**: Probability Density Function and Cumulative Distribution Function showing bandwidth distributions.
- **Figure 3 (ACF)**: Autocorrelation coefficient curves showing periodicity and self-similarity of IO traces.
- **Figure 4 (FFT)**: Fast Fourier Transform power spectrum identifying dominant frequency peaks.
- **Figure 5 (Burst Detection)**: Visual comparison of I/O bursts detected using the Adaptive Prediction Model vs. a fixed-threshold baseline ($P_{95}$).

---

## License
This project is licensed under the MIT License.
