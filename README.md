# Adaptive I/O Burst Characterization and Detection

Reproducible code for the causal event-level I/O-burst study based on Darshan POSIX/DXT traces.

## Methodological rules implemented in the code

- Detection uses DXT segment timing; POSIX aggregates are not expanded into invented operation timelines.
- A real temporal trace is accepted only when total DXT bytes agree with POSIX bytes within 5%.
- Bandwidth is regularized at 10 ms; DXT bytes are conserved by overlap-weighted binning.
- Every threshold at time `t` uses samples strictly before `t`.
- The detector uses aggregate bandwidth; I/O size remains a characterization variable.
- Candidate crossings are converted to events with a 3-of-5 persistence rule and one-bin-gap merging.
- Controlled accuracy uses independent injected events and one-to-one temporal-IoU matching.
- Real Darshan detections are descriptive only; they are not treated as contention labels.

## Files

```text
reproduce_article_figures.py   Darshan parsing, characterization, real-trace detection,
                               primary controlled benchmark and sensitivity helpers
extended_validation.py         IoU 0.30/0.50 evaluation, calibrated mu+k sigma baseline,
                               bootstrap CIs and amplitude-duration sweep
tests/test_detector.py         Causality, byte conservation and event-matching tests
requirements.txt               Python dependencies
pytest.ini                     Test import path
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Tests

```bash
python -m py_compile reproduce_article_figures.py extended_validation.py tests/test_detector.py
python -m pytest -q
```

Expected test result:

```text
6 passed
```

## Real traces and characterization

Place the Darshan logs in `logs/`, then run:

```bash
python reproduce_article_figures.py \
  --input-dir logs \
  --output-dir results/real \
  --synthetic-seeds 30 \
  --skip-sensitivity \
  --clean
```

The DXT/POSIX coverage check is written to `results/real/csv/dxt_coverage.csv`.
Only traces marked `dxt_complete=True` are used for real temporal detections.

For the trace set used in the experiments, the accepted temporal traces are NAMD, HACC and YOMBO.
E3SM has no DXT stream; LIFE-SCIENCE has incomplete DXT coverage; the IOR-HDF5 example has inconsistent/overlapping DXT coverage.

## Controlled event-level evaluation

Run the controlled evaluation with:

```bash
python extended_validation.py \
  --output-dir results/controlled \
  --seeds 30 \
  --sweep-seeds 6 \
  --sensitivity-seeds 30
```

Primary matching uses temporal IoU >= 0.30; IoU >= 0.50 is reported as a stricter criterion.
The sensitivity set uses seeds 1000--1029, disjoint from the primary benchmark, and evaluates:

- `W = {0.5, 1, 2, 4, 8, 15}` s at `tau=3`, `C_min=0.25`;
- `tau = {3, 3.5, 4}` at `W=2` s, `C_min=0.25`;
- `C_min = {0.20, 0.25, 0.30}` at `W=2` s, `tau=3`.

The calibrated `mu+k sigma` reference is fitted on an independent burst-free realization of the same background regime and then frozen on the test trace. Global P95 is retained as a simple non-causal reference.

## Scientific scope

Injected events provide detector-independent statistical labels; they do not prove storage saturation or application slowdown. Production validation requires independent system-level signals such as storage utilization, queueing or application slowdown.

The lightweight Darshan binary parser is included for reproducibility. Archival results should also be cross-checked with the official Darshan/PyDarshan toolchain.

Before redistributing Darshan logs, verify that paths, user/job metadata and trace redistribution rights are safe for public release.
