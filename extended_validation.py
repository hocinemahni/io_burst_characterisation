#!/usr/bin/env python3
"""Controlled event-level validation for the burst detector.

Includes calibrated mu+k*sigma, temporal-IoU matching, paired bootstrap
confidence intervals, amplitude-duration sweeps, and parameter sensitivity.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reproduce_article_figures as core

METHODS = [
    "global_p95", "saeedizade_style", "causal_mean_3sigma",
    "causal_hampel", "crad",
]
PRETTY = {
    "global_p95": "P95",
    "saeedizade_style": "Calibrated mu+k sigma",
    "causal_mean_3sigma": "Rolling mu+3 sigma",
    "causal_hampel": "Median/MAD",
    "crad": "Proposed",
}


def event_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return core.event_iou(a, b)


def event_metrics_iou(pred: np.ndarray, truth: np.ndarray, min_iou: float = 0.30,
                      max_gap_bins: int = 1) -> Dict[str, float]:
    return core.event_metrics(pred, truth, min_iou=min_iou, max_gap_bins=max_gap_bins)

def saeedizade_style_mask(signal: pd.DataFrame, persistence_window: int = 5,
                          persistence_hits: int = 3, calibration_bins: int = 300,
                          target_tail: float = 0.01,
                          calibration_signal: pd.DataFrame | None = None) -> Tuple[np.ndarray, float, float]:
    if calibration_signal is None:
        # Fallback calibration when no separate calibration signal is provided.
        x = signal["bandwidth_mb_s"].astype(float)
        ncal = min(calibration_bins, max(30, len(x)//5))
        calibration_signal = pd.DataFrame({
            "timestamp_s": signal["timestamp_s"].iloc[:ncal].to_numpy(),
            "bandwidth_mb_s": x.iloc[:ncal].to_numpy(),
        })
    return core.calibrated_mu_k_sigma_mask(
        signal, calibration_signal, persistence_window, persistence_hits, target_tail
    )

def detect_method(signal: pd.DataFrame, method: str, window: int, tau: float,
                  periodic_threshold: float, active_threshold: float,
                  periodic_update: int, persistence_window: int,
                  persistence_hits: int, calibration_signal: pd.DataFrame | None = None) -> Tuple[np.ndarray, Dict[str, float]]:
    if method == "crad":
        d = core.run_crad(signal, window=window, tau=tau,
                          periodic_threshold=periodic_threshold,
                          active_threshold=active_threshold,
                          periodic_update=periodic_update,
                          persistence_window=persistence_window,
                          persistence_hits=persistence_hits)
        return d["burst"].to_numpy(bool), {}
    if method == "saeedizade_style":
        mask, k, thr = saeedizade_style_mask(signal, persistence_window, persistence_hits, calibration_signal=calibration_signal)
        return mask, {"saeedizade_k": k, "saeedizade_threshold": thr}
    return core.run_baseline(signal, method, window, persistence_window, persistence_hits), {}


def evaluate(seeds: int, dt: float, window: int, tau: float,
             periodic_threshold: float, active_threshold: float,
             periodic_update: int, persistence_window: int,
             persistence_hits: int) -> pd.DataFrame:
    rows = []
    for scenario in ["stationary", "nonstationary", "periodic", "sparse"]:
        for seed in range(seeds):
            signal, truth = core.generate_synthetic_scenario(scenario, seed=seed, dt=dt)
            for method in METHODS:
                calibration_signal = None
                if method == "saeedizade_style":
                    bg = background(scenario, len(signal), seed + 100000)
                    calibration_signal = pd.DataFrame({"timestamp_s": np.arange(len(bg))*dt, "bandwidth_mb_s": bg})
                det, extra = detect_method(signal, method, window, tau, periodic_threshold,
                                           active_threshold, periodic_update,
                                           persistence_window, persistence_hits, calibration_signal)
                for iou in (0.30, 0.50):
                    m = event_metrics_iou(det, truth, min_iou=iou)
                    rows.append({"scenario":scenario,"seed":seed,"method":method,
                                 "iou_threshold":iou, **m, **extra})
    return pd.DataFrame(rows)


def bootstrap_pairs(df: pd.DataFrame, min_iou: float = 0.30, n_boot: int = 20000,
                    seed: int = 20260902) -> pd.DataFrame:
    d = df[np.isclose(df.iou_threshold, min_iou)]
    crad = d[d.method=="crad"][["scenario","seed","f1"]].rename(columns={"f1":"crad"})
    rows=[]; rng=np.random.default_rng(seed)
    for baseline in METHODS:
        if baseline == "crad": continue
        b = d[d.method==baseline][["scenario","seed","f1"]].rename(columns={"f1":"base"})
        z=crad.merge(b,on=["scenario","seed"]); diff=(z.crad-z.base).to_numpy()
        idx=rng.integers(0,len(diff),(n_boot,len(diff)))
        means=diff[idx].mean(axis=1)
        rows.append({"baseline":baseline,"mean_f1_difference":float(diff.mean()),
                     "ci95_low":float(np.quantile(means,.025)),
                     "ci95_high":float(np.quantile(means,.975)),
                     "fraction_crad_better":float((diff>0).mean()),
                     "fraction_ties":float((diff==0).mean()),"n_pairs":len(diff)})
    return pd.DataFrame(rows)


def background(kind: str, n: int, seed: int) -> np.ndarray:
    return core.generate_background(kind, n, seed)

def fixed_burst_scenario(kind: str, amplitude_factor: float, duration_bins: int,
                         seed: int, n: int=5000, dt: float=.01) -> Tuple[pd.DataFrame,np.ndarray]:
    x=background(kind,n,seed); truth=np.zeros(n,bool); rng=np.random.default_rng(seed+99173)
    starts=[]
    while len(starts)<20:
        st=int(rng.integers(300,n-duration_bins-1))
        if any(abs(st-s)<max(40,2*duration_bins) for s in starts): continue
        if kind=="nonstationary":
            bounds=[j*(n//5) for j in range(1,5)]
            if min(abs(st-b) for b in bounds)<100: continue
        starts.append(st)
        local=x[max(0,st-200):st]; pos=local[local>0]
        base=float(np.median(pos)) if len(pos) else 10.
        shape=np.hanning(duration_bins+2)[1:-1] if duration_bins>2 else np.ones(duration_bins)
        shape=.5+.5*shape
        x[st:st+duration_bins]+=base*amplitude_factor*shape
        truth[st:st+duration_bins]=True
    return pd.DataFrame({"timestamp_s":np.arange(n)*dt,"bandwidth_mb_s":x}), truth


def intensity_duration(seeds: int, dt: float, window: int, tau: float,
                       periodic_threshold: float, active_threshold: float,
                       periodic_update: int, persistence_window: int,
                       persistence_hits: int) -> pd.DataFrame:
    rows=[]
    amps=[1.5,2.0,2.5,3.0,4.0]; durations=[3,5,8,12]
    methods=["saeedizade_style","causal_mean_3sigma","causal_hampel","crad"]
    for amp in amps:
        for dur in durations:
            for scenario in ["stationary","nonstationary","periodic","sparse"]:
                for seed in range(2000,2000+seeds):
                    signal,truth=fixed_burst_scenario(scenario,amp,dur,seed,dt=dt)
                    for method in methods:
                        calibration_signal = None
                        if method == "saeedizade_style":
                            bg = background(scenario, len(signal), seed + 100000)
                            calibration_signal = pd.DataFrame({"timestamp_s": np.arange(len(bg))*dt, "bandwidth_mb_s": bg})
                        det,_=detect_method(signal,method,window,tau,periodic_threshold,
                                            active_threshold,periodic_update,
                                            persistence_window,persistence_hits, calibration_signal)
                        m=event_metrics_iou(det,truth,.30)
                        rows.append({"amplitude_factor":amp,"duration_bins":dur,
                                     "duration_ms":dur*dt*1000,"scenario":scenario,
                                     "seed":seed,"method":method,"f1":m["f1"]})
    return pd.DataFrame(rows)


def sensitivity(seeds: int, dt: float, persistence_window: int,
                persistence_hits: int) -> pd.DataFrame:
    configs = []
    configs += [(w, 3.0, 0.25) for w in [0.5, 1.0, 2.0, 4.0, 8.0, 15.0]]
    configs += [(2.0, tau, 0.25) for tau in [3.0, 3.5, 4.0]]
    configs += [(2.0, 3.0, cmin) for cmin in [0.20, 0.25, 0.30]]
    configs = list(dict.fromkeys(configs))

    rows=[]
    for hs, tau, pthr in configs:
        window=max(20,int(round(hs/dt)))
        f=[]
        for scenario in ["stationary","nonstationary","periodic","sparse"]:
            for seed in range(1000,1000+seeds):
                sig,tr=core.generate_synthetic_scenario(scenario,seed=seed,dt=dt)
                d=core.run_crad(sig,window=window,tau=tau,periodic_threshold=pthr,
                                persistence_window=persistence_window,
                                persistence_hits=persistence_hits)["burst"].to_numpy(bool)
                f.append(event_metrics_iou(d,tr,.30)["f1"])
        farr=np.asarray(f,dtype=float)
        rows.append({"history_s":hs,"window_bins":window,"tau":tau,
                     "periodic_threshold":pthr,"macro_event_f1":float(np.mean(farr)),
                     "standard_error":float(np.std(farr,ddof=1)/np.sqrt(len(farr))) if len(farr)>1 else np.nan,
                     "n_scenario_seed_pairs":int(len(farr))})
    return pd.DataFrame(rows)

def plot_f1(summary: pd.DataFrame, out: Path):
    s=summary[np.isclose(summary.iou_threshold,.30)]; scenarios=["stationary","nonstationary","periodic","sparse"]
    x=np.arange(4); width=.115
    fig,ax=plt.subplots(figsize=(8.7,3.7))
    for i,m in enumerate(METHODS):
        vals=[float(s[(s.method==m)&(s.scenario==sc)].f1_mean.iloc[0]) for sc in scenarios]
        ax.bar(x+(i-(len(METHODS)-1)/2)*width,vals,width,label=PRETTY[m])
    ax.set_xticks(x); ax.set_xticklabels(["Stationary","Non-stationary","Periodic","Sparse"])
    ax.set_ylim(0,1.03); ax.set_ylabel("Event-level F1 (IoU >= 0.30)"); ax.grid(axis="y",alpha=.25)
    ax.legend(fontsize=7,ncol=4); fig.tight_layout(); fig.savefig(out,dpi=220); plt.close(fig)


def plot_heatmaps(df: pd.DataFrame, out: Path):
    methods=["saeedizade_style","causal_mean_3sigma","causal_hampel","crad"]
    fig,axes=plt.subplots(1,4,figsize=(12.5,3.2),sharey=True)
    for ax,m in zip(axes,methods):
        z=df[df.method==m].groupby(["amplitude_factor","duration_ms"]).f1.mean().unstack("duration_ms")
        im=ax.imshow(z.to_numpy(),vmin=0,vmax=1,aspect="auto",origin="lower")
        ax.set_title(PRETTY[m],fontsize=9); ax.set_xticks(range(len(z.columns))); ax.set_xticklabels([f"{v:g}" for v in z.columns],fontsize=8)
        ax.set_yticks(range(len(z.index))); ax.set_yticklabels([f"{v:g}" for v in z.index],fontsize=8)
        ax.set_xlabel("Duration (ms)")
        for i in range(z.shape[0]):
            for j in range(z.shape[1]): ax.text(j,i,f"{z.iloc[i,j]:.2f}",ha="center",va="center",fontsize=6)
    axes[0].set_ylabel("Amplitude factor"); fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label="Macro event F1")
    fig.subplots_adjust(left=.06,right=.94,bottom=.18,top=.88,wspace=.15); fig.savefig(out,dpi=220); plt.close(fig)


def plot_controlled_example(dt: float, window: int, tau: float, pthr: float, athr: float,
                            pup: int, pw: int, ph: int, out: Path):
    sig,tr=fixed_burst_scenario("stationary",2.5,8,2222,dt=dt)
    cr=core.run_crad(sig,window=window,tau=tau,periodic_threshold=pthr,active_threshold=athr,
                     periodic_update=pup,persistence_window=pw,persistence_hits=ph)
    bg=background("stationary",len(sig),102222); cal=pd.DataFrame({"timestamp_s":np.arange(len(bg))*dt,"bandwidth_mb_s":bg}); smask,k,sthr=saeedizade_style_mask(sig,pw,ph,calibration_signal=cal)
    ev=core.mask_to_events(tr,0)[0]; a=max(0,ev[0]-80); b=min(len(sig),ev[1]+100)
    fig,ax=plt.subplots(figsize=(8.3,3.2))
    ax.plot(sig.timestamp_s.iloc[a:b],sig.bandwidth_mb_s.iloc[a:b],lw=1,label="Bandwidth")
    ax.plot(sig.timestamp_s.iloc[a:b],cr.threshold_crad.iloc[a:b],ls="--",lw=1,label="Proposed threshold")
    ax.axhline(sthr,ls=":",lw=1,label=f"Calibrated mu+k sigma (k={k:.2f})")
    ax.axvspan(sig.timestamp_s.iloc[ev[0]],sig.timestamp_s.iloc[ev[1]],alpha=.18,label="Injected ground-truth event")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Bandwidth (MiB/s)"); ax.grid(alpha=.25); ax.legend(fontsize=7,ncol=2)
    fig.tight_layout(); fig.savefig(out,dpi=220); plt.close(fig)


def write_tables(summary: pd.DataFrame, boot: pd.DataFrame, sens: pd.DataFrame, out: Path):
    out.mkdir(parents=True,exist_ok=True)
    for iou,name in [(.30,"synthetic_f1_iou30.tex"),(.50,"synthetic_f1_iou50.tex")]:
        s=summary[np.isclose(summary.iou_threshold,iou)]; piv=s.pivot(index="method",columns="scenario",values="f1_mean")
        lines=[r"\begin{tabular}{lrrrrr}",r"\toprule",r"Method & Stationary & Non-stat. & Periodic & Sparse & Macro \\",r"\midrule"]
        latex_names={"global_p95":"P95","saeedizade_style":r"Calibrated $\mu+k\sigma$","causal_mean_3sigma":r"Rolling $\mu+3\sigma$","causal_hampel":"Median/MAD","crad":r"\textbf{Proposed}"}
        for m in METHODS:
            vals=[float(piv.loc[m,sc]) for sc in ["stationary","nonstationary","periodic","sparse"]]; macro=np.mean(vals)
            vals_s=" & ".join(f"{v:.3f}" for v in vals)
            lines.append(f"{latex_names[m]} & {vals_s} & {macro:.3f} \\\\")
        lines += [r"\bottomrule",r"\end{tabular}"]
        (out/name).write_text("\n".join(lines)+"\n")
    lines=[r"\begin{tabular}{lrrr}",r"\toprule",r"Baseline & $\Delta$F1 & 95\% CI low & 95\% CI high \\",r"\midrule"]
    for _,r in boot.iterrows():
        lines.append(f"{PRETTY[r.baseline]} & {r.mean_f1_difference:+.3f} & {r.ci95_low:+.3f} & {r.ci95_high:+.3f} \\\\")
    lines += [r"\bottomrule",r"\end{tabular}"]; (out/"bootstrap_updated.tex").write_text("\n".join(lines)+"\n")
    best=sens.sort_values("macro_event_f1",ascending=False).iloc[0]
    chosen=sens[(sens.history_s==2.)&(sens.tau==3.)&np.isclose(sens.periodic_threshold,.25)].iloc[0]
    txt="\n".join([r"\begin{tabular}{lrrrr}",r"\toprule",r"Configuration & History (s) & $\tau$ & $C_{\min}$ & Macro F1 \\",r"\midrule",f"Chosen & {chosen.history_s:.1f} & {chosen.tau:.1f} & {chosen.periodic_threshold:.2f} & {chosen.macro_event_f1:.3f} \\\\",f"Best tested point & {best.history_s:.1f} & {best.tau:.1f} & {best.periodic_threshold:.2f} & {best.macro_event_f1:.3f} \\\\",r"\bottomrule",r"\end{tabular}"])+"\n"
    (out/"sensitivity_updated.tex").write_text(txt)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,default=Path("results_updated")); ap.add_argument("--seeds",type=int,default=30); ap.add_argument("--sweep-seeds",type=int,default=6); ap.add_argument("--sensitivity-seeds",type=int,default=30)
    a=ap.parse_args(); out=a.output_dir; (out/"csv").mkdir(parents=True,exist_ok=True); (out/"figures").mkdir(parents=True,exist_ok=True)
    dt=.01; window=200; tau=3.; pthr=.25; athr=.20; pup=20; pw=5; ph=3
    df=evaluate(a.seeds,dt,window,tau,pthr,athr,pup,pw,ph); df.to_csv(out/"csv"/"synthetic_event_metrics_all_runs_iou.csv",index=False)
    summary=df.groupby(["iou_threshold","scenario","method"],as_index=False).agg(precision_mean=("precision","mean"),recall_mean=("recall","mean"),f1_mean=("f1","mean"),f1_std=("f1","std"),delay_bins_mean=("mean_detection_delay_bins","mean"),matched_iou_mean=("mean_matched_iou","mean"))
    summary.to_csv(out/"csv"/"synthetic_event_metrics_summary_iou.csv",index=False)
    boot=bootstrap_pairs(df); boot.to_csv(out/"csv"/"paired_f1_bootstrap_updated.csv",index=False)
    sweep=intensity_duration(a.sweep_seeds,dt,window,tau,pthr,athr,pup,pw,ph); sweep.to_csv(out/"csv"/"intensity_duration_metrics.csv",index=False)
    sens=sensitivity(a.sensitivity_seeds,dt,pw,ph); sens.to_csv(out/"csv"/"crad_sensitivity_iou30.csv",index=False)
    plot_f1(summary,out/"figures"/"synthetic_event_f1_updated.png"); plot_heatmaps(sweep,out/"figures"/"intensity_duration_heatmaps.png"); plot_controlled_example(dt,window,tau,pthr,athr,pup,pw,ph,out/"figures"/"controlled_burst_example.png")
    # History-window sensitivity with 95% confidence intervals.
    d=sens[(np.isclose(sens.periodic_threshold,.25)) & (np.isclose(sens.tau,3.0))].sort_values("history_s")
    fig,ax=plt.subplots(figsize=(5.8,3.4))
    yerr=1.96*d.standard_error.to_numpy(float) if "standard_error" in d else None
    ax.errorbar(d.history_s,d.macro_event_f1,yerr=yerr,marker="o",capsize=3)
    ax.set_xlabel("History length (s)"); ax.set_ylabel("Macro event F1 (IoU >= 0.30)"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/"figures"/"window_sensitivity.png",dpi=220); plt.close(fig)
    write_tables(summary,boot,sens,out/"tables")
    print(summary[np.isclose(summary.iou_threshold,.30)].pivot(index="method",columns="scenario",values="f1_mean"))
    print("\nMacro IoU .30:\n",summary[np.isclose(summary.iou_threshold,.30)].groupby("method").f1_mean.mean().sort_values(ascending=False))
    print("\nMacro IoU .50:\n",summary[np.isclose(summary.iou_threshold,.50)].groupby("method").f1_mean.mean().sort_values(ascending=False))
    print("\nBootstrap:\n",boot)

if __name__=="__main__": main()
