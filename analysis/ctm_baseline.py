#!/usr/bin/env python3
"""
===============================================================================
PHASE 3E — CELL TRANSMISSION MODEL (CTM / LWR) PHYSICAL BASELINE  (EIC-2)
===============================================================================

Editor-in-Chief, comment 2 (verbatim, revision_email.md:11):
  "Any work about traffic forecasting must compare results with model-based
   approaches that entail the physical laws behind traffic, especially highway
   traffic."

This is a GENUINE physics-based baseline — NOT a statistical model. It implements
the Cell Transmission Model (Daganzo 1994), the standard discretization of the
Lighthill–Whitham–Richards (LWR) macroscopic traffic-flow theory, on the sensor
road network, using a Greenshields fundamental diagram.

PHYSICAL LAWS USED
------------------
1. Conservation of vehicles (continuity):  n_i(t+1) = n_i(t) + C·(inflow_i − outflow_i)
2. Fundamental diagram (Greenshields):     v = v_f·(1 − n),   q(n) = n·(1 − n)
   where n = ρ/ρ_jam is normalized density in [0,1].
3. Godunov / CTM flux (demand–supply):
       Demand  D(n) = q(n) if n ≤ n_c else q_max     (sending function)
       Supply  S(n) = q_max if n ≤ n_c else q(n)     (receiving function)
       edge flux f(i→j) = min( D_i·outsplit_ij , S_j·insplit_ij )
   with n_c = 0.5, q_max = 0.25 (Greenshields capacity).
4. Spatial propagation on the DIRECTED road graph (adj_mx) → backward congestion
   waves emerge naturally from the supply constraint (this is the physics the EIC
   means by "system dynamics", absent in time-series baselines).

DATA NOTE
---------
METR-LA / PEMS-BAY provide only point SPEEDS (no density/flow). Following standard
physics-informed practice, we invert the Greenshields FD to get density from speed:
       n = clip(1 − v/v_f, 0, 1),
with per-sensor free-flow speed v_f calibrated from the TRAINING set only
(95th percentile of observed speed). This is disclosed as a modeling assumption.

FORECAST PROTOCOL (strictly causal, fair vs the DL models)
----------------------------------------------------------
At each test origin t: initialise density from the observed speed at t, then run
the CTM forward Q steps (no future data used). Convert predicted density back to
speed. Metrics are AVERAGED over steps 1..Q (same convention as the DL models'
evaluate()), MAPE masked at <5 mph, on the last-20% test split.

Output: results/ctm_baseline_results.csv
        columns: Model | Dataset | Q | MAE | RMSE | MAPE | N

Run:  python revision/phase3E_ctm_baseline.py
      python revision/phase3E_ctm_baseline.py --max-origins 500   # quick sanity
===============================================================================
"""

import argparse
import json
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
OUTPUT_PATH = ROOT / "results" / "ctm_baseline_results.csv"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
SEQ_LEN     = 12          # history length (to align origins with the DL models)
HORIZONS    = [3, 6, 12]
MAPE_MIN    = 5.0         # mph, MAPE mask (matches manuscript)
N_C         = 0.5         # critical normalized density (Greenshields)
Q_MAX       = 0.25        # capacity q(0.5)=0.25
COURANT     = 0.5         # stable flux multiplier per 5-min step (CFL-safe with clipping)
ADJ_THRESH  = 0.0         # keep all positive directed edges (diagonal removed)


def load_dataset(ds):
    d = DATA_ROOT / ds
    speed_norm = np.load(d / "scaled_speed.npy").astype(np.float64)        # (T,N)
    with open(d / "normalization_params.json") as f:
        p = json.load(f)
    mph = speed_norm * float(p["speed_std"]) + float(p["speed_mean"])
    mph = np.clip(mph, 0.0, None)                                          # remove negative-speed noise
    with open(d / "adj_mx.pkl", "rb") as f:
        _, _, A = pickle.load(f, encoding="latin1")
    A = np.array(A, dtype=np.float64)
    np.fill_diagonal(A, 0.0)                                               # self-loops don't transport mass
    return mph, A, p


def build_edges(A):
    """Directed edges with conservative split fractions."""
    src, dst = np.nonzero(A > ADJ_THRESH)            # i->j with weight A[i,j]
    w = A[src, dst]
    N = A.shape[0]
    out_tot = np.array(A.sum(axis=1)).ravel()        # i's total outgoing weight
    in_tot  = np.array(A.sum(axis=0)).ravel()        # j's total incoming weight
    out_split = w / np.maximum(out_tot[src], 1e-9)   # share of i's outflow to j
    in_split  = w / np.maximum(in_tot[dst], 1e-9)    # share of j's inflow from i
    # incidence matrices for fast scatter (N x E)
    E = len(src)
    M_src = sparse.csr_matrix((np.ones(E), (src, np.arange(E))), shape=(N, E))
    M_dst = sparse.csr_matrix((np.ones(E), (dst, np.arange(E))), shape=(N, E))
    is_source = (in_tot <= 1e-9)                      # cells with no upstream (network inlets)
    return src, dst, out_split, in_split, M_src, M_dst, is_source


def fd_speed_to_density(v, vf):
    return np.clip(1.0 - v / np.maximum(vf, 1e-6), 0.0, 1.0)

def fd_density_to_speed(n, vf):
    return vf * (1.0 - n)

def demand(n):
    q = n * (1.0 - n)
    return np.where(n <= N_C, q, Q_MAX)

def supply(n):
    q = n * (1.0 - n)
    return np.where(n <= N_C, Q_MAX, q)


def ctm_step(n, vf, edges):
    """One CTM/Godunov update. n: (R,N) batched over origins."""
    src, dst, out_split, in_split, M_src, M_dst, is_source = edges
    D = demand(n)                                     # (R,N)
    S = supply(n)                                     # (R,N)
    f = np.minimum(D[:, src] * out_split[None, :],
                   S[:, dst] * in_split[None, :])     # (R,E) edge flux
    outflow = f @ M_src.T                              # (R,N)  sum of fluxes leaving i
    inflow  = f @ M_dst.T                              # (R,N)  sum of fluxes entering j
    # network inlets keep a steady external inflow = their own demand (persistence of boundary)
    inflow = inflow + is_source[None, :] * D
    n_new = np.clip(n + COURANT * (inflow - outflow), 0.0, 1.0)
    return n_new


def run_dataset(ds, max_origins=None):
    print(f"\n{'='*70}\nCTM (LWR) PHYSICAL BASELINE  ·  {ds}\n{'='*70}")
    mph, A, p = load_dataset(ds)
    T, N = mph.shape
    n_train = int(TRAIN_RATIO * T)
    n_val   = int(VAL_RATIO * T)
    test_start = n_train + n_val

    # free-flow speed per sensor from TRAINING only (95th pctl)
    vf = np.percentile(mph[:n_train], 95, axis=0)
    vf = np.maximum(vf, 5.0)                          # guard tiny vf
    print(f"   sensors={N}  v_f mean={vf.mean():.1f} mph  test_steps={T-test_start}")

    edges = build_edges(A)
    print(f"   directed edges={len(edges[0])}  network inlets={int(edges[6].sum())}")

    # origins: each test time t with a full Q-window ahead
    max_Q = max(HORIZONS)
    origins = np.arange(test_start + SEQ_LEN, T - max_Q)
    if max_origins:
        origins = origins[:max_origins]
    R = len(origins)
    print(f"   forecast origins={R}")

    # initial density at each origin (from observed speed at t)
    v0 = mph[origins]                                 # (R,N)
    n = fd_density_to_speed  # placeholder to avoid lint; not used
    n_state = fd_speed_to_density(v0, vf)             # (R,N)

    # roll forward, capturing predictions at each horizon step
    preds_by_h = {}
    cur = n_state.copy()
    for step in range(1, max_Q + 1):
        cur = ctm_step(cur, vf, edges)
        if step in HORIZONS or step <= max(HORIZONS):
            preds_by_h[step] = fd_density_to_speed(cur, vf)   # (R,N) predicted speed at t+step

    results = []
    for Q in HORIZONS:
        # pool steps 1..Q (matches DL averaging over the horizon)
        yt_list, yp_list = [], []
        for k in range(1, Q + 1):
            yp_list.append(preds_by_h[k])                       # (R,N)
            yt_list.append(mph[origins + k])                    # (R,N) ground truth
        yp = np.concatenate(yp_list).ravel()
        yt = np.concatenate(yt_list).ravel()
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        m = yt >= MAPE_MIN
        mape = float(100.0 * np.mean(np.abs(yt[m] - yp[m]) / yt[m])) if m.sum() else float('nan')
        print(f"   Q={Q:2d}  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%  N={yt.size}  (avg steps 1..{Q})")
        results.append({"Model": "CTM-LWR", "Dataset": ds, "Q": Q,
                        "MAE": round(mae, 6), "RMSE": round(rmse, 6),
                        "MAPE": round(mape, 6), "N": int(yt.size)})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-origins", type=int, default=None,
                    help="limit forecast origins for a quick sanity run")
    args = ap.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in ["metr-la", "pems-bay"]:
        if not (DATA_ROOT / ds / "scaled_speed.npy").exists():
            print(f"skip {ds}: no data"); continue
        rows += run_dataset(ds, max_origins=args.max_origins)

    df = pd.DataFrame(rows)
    if args.max_origins:
        print("\n[QUICK SANITY RUN — not saved]")
        print(df.to_string(index=False))
        return 0
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n{'='*70}\nCTM-LWR PHYSICAL BASELINE — SUMMARY\n{'='*70}")
    print(df.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
