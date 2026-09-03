# Adaptive I/O Burst Characterization and Detection

Reproducible code for causal event-level I/O-burst characterization and detection from Darshan POSIX/DXT traces.

## Method

- Detection uses DXT segment timing; POSIX aggregates are not expanded into operation timelines.
- A temporal trace is accepted only when total DXT bytes agree with POSIX bytes within 5%.
- Bandwidth is regularized at 10 ms using overlap-weighted binning.
- Every threshold at time `t` uses only samples preceding `t`.
- Detection operates on aggregate bandwidth; I/O size is used for characterization.
- Candidate crossings are converted to events with a 3-of-5 persistence rule and one-bin-gap merging.
- Controlled evaluation uses independent injected events and one-to-one temporal-IoU matching.

## Repository files

```text
reproduce_article_figures.py   Darshan parsing, characterization and real-trace detection
extended_validation.py         Controlled evaluation, baselines, bootstrap and stress tests
tests/test_detector.py         Causality, byte-conservation and event-matching tests
requirements.txt               Python dependencies
pytest.ini                     Pytest configuration
Dockerfile                     Container image for tests and experiments
compose.yaml                   Docker Compose commands for tests and experiments
```

## Linux installation

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip

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
python -m py_compile reproduce_article_figures.py extended_validation.py tests/test_detector.py
python -m pytest -q
```

Expected result:

```text
6 passed
```

## Windows installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
```

## Real traces and characterization

Place Darshan logs in `logs/` and run:

```bash
python reproduce_article_figures.py \
  --input-dir logs \
  --output-dir results/real \
  --synthetic-seeds 30 \
  --skip-sensitivity \
  --clean
```

The DXT/POSIX coverage check is written to `results/real/csv/dxt_coverage.csv`. Only traces marked `dxt_complete=True` are used for temporal detection.

For the trace set used in the experiments, NAMD, HACC and YOMBO pass the temporal-coverage check. E3SM has no DXT stream; LIFE-SCIENCE has incomplete DXT coverage; the IOR-HDF5 example has inconsistent/overlapping DXT coverage.

## Controlled event-level evaluation

```bash
python extended_validation.py \
  --output-dir results/controlled \
  --seeds 30 \
  --sweep-seeds 6 \
  --sensitivity-seeds 30
```

Primary matching uses temporal IoU >= 0.30; IoU >= 0.50 is also reported. The sensitivity set uses seeds 1000--1029, disjoint from the primary benchmark, and evaluates:

- `W = {0.5, 1, 2, 4, 8, 15}` s at `tau=3`, `C_min=0.25`;
- `tau = {3, 3.5, 4}` at `W=2` s, `C_min=0.25`;
- `C_min = {0.20, 0.25, 0.30}` at `W=2` s, `tau=3`.

The calibrated `mu+k sigma` baseline is fitted on an independent burst-free realization of the same background regime and then kept fixed on the test trace. Global P95 is retained as a simple non-causal reference.

## Docker

Build the image:

```bash
docker build -t io-burst .
```

Run the test suite inside the container:

```bash
docker run --rm io-burst
```

Run the controlled evaluation and write results to the local `results/` directory:

```bash
mkdir -p results

docker run --rm \
  -v "$(pwd)/results:/app/results" \
  io-burst \
  python extended_validation.py \
    --output-dir /app/results/controlled \
    --seeds 30 \
    --sweep-seeds 6 \
    --sensitivity-seeds 30
```

Run the Darshan analysis with local logs mounted read-only:

```bash
mkdir -p logs results

docker run --rm \
  -v "$(pwd)/logs:/app/logs:ro" \
  -v "$(pwd)/results:/app/results" \
  io-burst \
  python reproduce_article_figures.py \
    --input-dir /app/logs \
    --output-dir /app/results/real \
    --synthetic-seeds 30 \
    --skip-sensitivity \
    --clean
```

### Docker Compose

The same workflows are available through `compose.yaml`:

```bash
docker compose build
docker compose run --rm tests
docker compose run --rm controlled
docker compose run --rm real
```

The `real` service expects Darshan files in `./logs/`. Generated files are written under `./results/`.
