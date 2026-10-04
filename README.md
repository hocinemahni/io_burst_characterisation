# RAPID: Regime-Adaptive Persistent I/O Detection

Reproducible code and controlled validation for **RAPID**, a causal regime-aware detector of persistent high-bandwidth I/O events from Darshan POSIX/DXT traces.

## What changed in the revised evaluation

The controlled benchmark still contains four 50-s regimes sampled at 10 ms (stationary, non-stationary, periodic, sparse), 30 seeds per regime, and 20 independently injected events per trace: **2,400 labeled events** in total. The detector is evaluated with one-to-one temporal-IoU matching at `IoU >= 0.30` and `IoU >= 0.50`.

The main comparison now distinguishes generic statistical references from HPC I/O literature methods:

- **P95**: non-causal global percentile reference, followed by the common persistence/event-formation rule.
- **Median/MAD**: causal robust rolling reference, followed by the common persistence/event-formation rule.
- **Saeedizade et al. (e-Science 2023)**: the published global `mu + k*sigma` burst-labeling principle. A fixed regime-level threshold is selected so that about 1% of pooled bins exceed it. No RAPID persistence is added. This reproduces the labeling rule, **not** the downstream ML predictor.
- **IOSI-WT (FAST 2014)**: the documented per-trace wavelet burst-isolation stage (background-noise subtraction, discrete Meyer level 2, trough filtering). The later cross-run CLIQUE signature-extraction stage is not applied because injected event positions are randomized across repetitions.
- **FTIO v0.0.9**: FTIO periodicity detection combined with the current burst-width routine; within each detected period, the shortest contiguous interval containing 95% of `BW^2` energy is returned. This burst-width feature is newer than the original IPDPS'24 paper.
- **RAPID**: causal regime-aware threshold selection followed by a 3-of-5 persistence rule.

The literature-specific F1 values for IOSI-WT and FTIO on randomized injected events are **scope diagnostics**, not claims that RAPID universally outperforms the complete IOSI or FTIO systems. Full IOSI assumes repeatable bursts across executions; FTIO assumes periodic I/O phases.

## Main controlled results

At temporal `IoU >= 0.30`, macro event F1 is:

| Method | Macro F1 |
|---|---:|
| RAPID | **0.825** |
| Median/MAD | 0.745 |
| P95 | 0.689 |
| Saeedizade global `mu+k*sigma` | 0.679 |
| IOSI-WT | 0.067 |
| FTIO v0.0.9 burst-width | 0.012 |

At `IoU >= 0.50`:

| Method | Macro F1 |
|---|---:|
| RAPID | **0.594** |
| Median/MAD | 0.538 |
| Saeedizade global `mu+k*sigma` | 0.515 |
| P95 | 0.496 |
| IOSI-WT | 0.031 |
| FTIO v0.0.9 burst-width | 0.009 |

Paired bootstrap differences at `IoU >= 0.30` are reported only against the directly comparable threshold-based references:

- RAPID - P95: `+0.136`, 95% CI `[+0.114, +0.157]`
- RAPID - Saeedizade: `+0.146`, 95% CI `[+0.123, +0.169]`
- RAPID - Median/MAD: `+0.080`, 95% CI `[+0.061, +0.101]`

## Repository files

```text
reproduce_article_figures.py   Darshan parsing, characterization and real-trace RAPID detection
extended_validation.py         Controlled evaluation + literature references + bootstrap + stress test
tests/test_detector.py         Core causality, byte-conservation and event-matching tests
tests/test_literature_methods.py
                               Saeedizade / IOSI-WT / FTIO validation tests
requirements.txt               Python dependencies
pytest.ini                     Pytest configuration
Dockerfile                     Container image for tests and experiments
compose.yaml                   Docker Compose commands
```

## Installation

```bash
git clone https://github.com/hocinemahni/io_burst_characterisation.git
cd io_burst_characterisation

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
```

Run the tests:

```bash
python -m py_compile reproduce_article_figures.py extended_validation.py tests/test_detector.py tests/test_literature_methods.py
python -m pytest -q
```

## Real Darshan/DXT traces

Place Darshan logs in `logs/` and run:

```bash
python reproduce_article_figures.py \
  --input-dir logs \
  --output-dir results/real \
  --synthetic-seeds 30 \
  --skip-sensitivity \
  --clean
```

Temporal analysis uses **DXT segments only**. DXT byte totals are checked against POSIX totals and accepted when coverage is within 5%. In the current artifact, NAMD, HACC and YOMBO pass this temporal-coverage check. Real detections are descriptive because the logs do not contain independent contention/saturation/slowdown labels.

## Revised controlled evaluation

```bash
python extended_validation.py \
  --output-dir results/controlled \
  --seeds 30 \
  --sweep-seeds 6 \
  --sensitivity-seeds 30
```

This generates:

```text
results/controlled/csv/literature_event_metrics_all_runs.csv
results/controlled/csv/literature_event_per_regime.csv
results/controlled/csv/literature_event_summary_iou030_050.csv
results/controlled/csv/literature_method_parameters.csv
results/controlled/csv/bootstrap_updated.csv
results/controlled/csv/stress_test_all_runs.csv
results/controlled/csv/stress_test_summary.csv
results/controlled/csv/rapid_sensitivity.csv
results/controlled/figures/synthetic_event_f1_updated.png
results/controlled/figures/intensity_duration_heatmaps.png
results/controlled/tables/*.tex
```

The amplitude-duration study uses factors `{1.5, 2, 2.5, 3, 4}` and durations `{30, 50, 80, 120}` ms. At amplitude `2.5x`, RAPID reaches approximately `0.017 / 0.930 / 0.941 / 0.896` for `30 / 50 / 80 / 120` ms. This makes the expected limitation of the 3-of-5 persistence rule on 30-ms events explicit.

## RAPID parameters

The primary configuration is:

```text
sampling interval       10 ms
history W               2 s / 200 bins
robust tau               3
robust quantile floor   Q0.90
active fraction switch  0.20
spectral concentration  0.25
FFT refresh             20 bins
persistence             3 of 5 bins
```

Sensitivity is evaluated on seeds `1000--1029`, disjoint from the primary benchmark.

## Interpretation

Synthetic injections provide exact temporal ground truth for detector evaluation; they are not a universal definition of harmful storage contention. A production validation should correlate detections with independent signals such as OSS load, queueing, application slowdown, or interference from co-running jobs.
