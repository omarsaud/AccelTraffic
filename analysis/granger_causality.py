"""
Fast Granger causality (acceleration -> speed) on ALL sensors (207 METR, 325 PEMS),
matching the MI panel's coverage. Uses direct least-squares for the ssr F-test
(mathematically identical to statsmodels' ssr_ftest) so all sensors run in ~1-2 min.
Same data variants as the MI/Granger analyses: METR=metr-la-v2, PEMS=pems-bay.
Output: results/granger_causality_allsensors.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent.parent / "results" / "granger_causality_allsensors.csv"
VARIANT = {"metr-la": "metr-la-v2", "pems-bay": "pems-bay"}
MAXLAG = 12
TRAIN_RATIO = 0.70


def rss(X, y):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r)


def granger_F(y, x, L):
    """ssr-based Granger F: does past x improve AR(L) model of y?"""
    T = len(y)
    n = T - L
    if n <= 2 * L + 2:
        return np.nan, np.nan
    yt = y[L:]
    cols_r = [np.ones(n)] + [y[L - j: T - j] for j in range(1, L + 1)]
    Xr = np.column_stack(cols_r)
    cols_u = cols_r + [x[L - j: T - j] for j in range(1, L + 1)]
    Xu = np.column_stack(cols_u)
    rss_r, rss_u = rss(Xr, yt), rss(Xu, yt)
    if rss_u <= 0:
        return np.nan, np.nan
    df2 = n - (2 * L + 1)
    F = ((rss_r - rss_u) / L) / (rss_u / df2)
    p = float(stats.f.sf(F, L, df2))
    return float(F), p


def run_dataset(ds):
    vdir = VARIANT[ds]
    speed = np.load(DATA / vdir / "scaled_speed.npy").astype(np.float64)
    accel = np.load(DATA / vdir / "scaled_acceleration.npy").astype(np.float64)
    T, N = speed.shape
    tr = int(TRAIN_RATIO * T)
    speed, accel = speed[:tr], accel[:tr]
    per_lag = {L: [] for L in range(1, MAXLAG + 1)}
    ok = 0
    for s in range(N):
        y, x = speed[:, s], accel[:, s]
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
            continue
        if np.std(y) < 1e-6 or np.std(x) < 1e-6:
            continue
        for L in range(1, MAXLAG + 1):
            F, p = granger_F(y, x, L)
            if np.isfinite(F):
                per_lag[L].append((F, p))
        ok += 1
    rows = []
    for L in range(1, MAXLAG + 1):
        vals = per_lag[L]
        if not vals:
            continue
        F = np.array([v[0] for v in vals]); p = np.array([v[1] for v in vals])
        rows.append(dict(dataset=ds, lag=L, lag_minutes=L * 5, n_sensors=len(vals),
                         median_F=float(np.median(F)), q25_F=float(np.percentile(F, 25)),
                         q75_F=float(np.percentile(F, 75)), mean_F=float(np.mean(F)),
                         pct_sig_05=float(np.mean(p < 0.05) * 100),
                         pct_sig_001=float(np.mean(p < 0.001) * 100)))
    print(f"  {ds} ({vdir}): {ok}/{N} sensors", flush=True)
    return rows


def main():
    allrows = []
    for ds in ["metr-la", "pems-bay"]:
        print(f"Granger (fast, ALL sensors): {ds} ...", flush=True)
        allrows += run_dataset(ds)
    df = pd.DataFrame(allrows)
    df.to_csv(OUT, index=False)
    print(f"[OK] {OUT}", flush=True)
    for ds in ["metr-la", "pems-bay"]:
        d = df[df.dataset == ds].sort_values("lag")
        print(f"{ds}: medianF {d.median_F.iloc[0]:.0f}->{d.median_F.iloc[-1]:.1f}; "
              f"%sig(.001) {d.pct_sig_001.iloc[0]:.0f}%->{d.pct_sig_001.iloc[-1]:.0f}%; n={int(d.n_sensors.iloc[0])}", flush=True)


if __name__ == "__main__":
    main()
