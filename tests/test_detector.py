import numpy as np
import pandas as pd

import reproduce_article_figures as R


def signal(x, dt=0.01):
    x = np.asarray(x, dtype=float)
    return pd.DataFrame({"timestamp_s": np.arange(len(x)) * dt, "bandwidth_mb_s": x})


def test_no_activity_no_burst():
    s = signal(np.zeros(1000))
    d = R.run_crad(s, window=200)
    assert not d["burst"].any()


def test_causality_future_change_does_not_change_past():
    rng = np.random.default_rng(4)
    x = 100 + rng.normal(0, 3, 1200)
    x[700:706] += 300
    s1 = signal(x)
    d1 = R.run_crad(s1, window=200)

    x2 = x.copy()
    x2[900:] += 10_000  # modify only the future
    d2 = R.run_crad(signal(x2), window=200)

    assert np.array_equal(d1.loc[:899, "candidate"].to_numpy(), d2.loc[:899, "candidate"].to_numpy())
    assert np.allclose(
        d1.loc[:899, "threshold_crad"].to_numpy(),
        d2.loc[:899, "threshold_crad"].to_numpy(),
        equal_nan=True,
    )


def test_injected_burst_is_detected():
    x = np.full(1000, 100.0)
    x[600:610] = 600.0
    d = R.run_crad(signal(x), window=200)
    assert d.loc[600:615, "burst"].any()


def test_regular_dxt_conserves_bytes():
    dxt = pd.DataFrame({
        "io_type": ["write", "read"],
        "start_time": [0.000, 0.015],
        "end_time": [0.020, 0.025],
        "length_bytes": [10_000_000, 5_000_000],
    })
    s = R.dxt_to_regular_bandwidth(dxt, bin_width_s=0.01)
    reconstructed = float((s["volume_mb"].sum()) * R.MB)
    assert abs(reconstructed - 15_000_000) < 1e-6


def test_iou_event_matching_strict():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import extended_validation as ev
    truth = np.zeros(20, dtype=bool); truth[5:10] = True
    pred = np.zeros(20, dtype=bool); pred[8:13] = True
    assert ev.event_metrics_iou(pred, truth, min_iou=0.20)["tp_events"] == 1
    assert ev.event_metrics_iou(pred, truth, min_iou=0.50)["tp_events"] == 0


def test_saeedizade_style_calibration_finite():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import extended_validation as ev
    sig, _ = R.generate_synthetic_scenario("stationary", n=5000, seed=7, dt=0.01)
    mask, k, threshold = ev.saeedizade_style_mask(sig)
    assert len(mask) == len(sig)
    assert np.isfinite(k) and k >= 0
    assert np.isfinite(threshold)
