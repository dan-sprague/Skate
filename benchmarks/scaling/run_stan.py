#!/usr/bin/env python3
"""Run Stan benchmark on the survival frailty model at varying hospital counts.
Uses cmdstanpy for parallel chains (matching PhaseSkate's parallel execution).
"""

import json
import math
import time
import os
import numpy as np
from pathlib import Path
from cmdstanpy import CmdStanModel
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).parent
STAN_MODEL = SCRIPT_DIR / "survival.stan"
DATA_DIR = SCRIPT_DIR / "data"

NUM_SAMPLES = 2000
NUM_WARMUP = 1000
NUM_CHAINS = 4
MAX_DEPTH = 10

# ── Fix TBB library conflict (miniforge shadows CmdStan's TBB) ──
import cmdstanpy
cmdstan_path = cmdstanpy.cmdstan_path()
tbb_lib = os.path.join(cmdstan_path, "stan", "lib", "stan_math", "lib", "tbb")
if "DYLD_LIBRARY_PATH" in os.environ:
    os.environ["DYLD_LIBRARY_PATH"] = tbb_lib + ":" + os.environ["DYLD_LIBRARY_PATH"]
else:
    os.environ["DYLD_LIBRARY_PATH"] = tbb_lib

# ── ESS computation (matching Stan's algorithm exactly) ──

def autocovariance(x):
    N = len(x)
    m = np.mean(x)
    xc = x - m
    acov = np.zeros(N)
    for lag in range(N):
        acov[lag] = np.dot(xc[:N-lag], xc[lag:]) / N
    return acov

def ess_func(chains_matrix):
    N, M = chains_matrix.shape
    if N < 4:
        return float(N * M)
    acov_all = []
    chain_var = np.zeros(M)
    chain_mean = np.zeros(M)
    for j in range(M):
        ac = autocovariance(chains_matrix[:, j])
        acov_all.append(ac)
        chain_var[j] = ac[0] * N / (N - 1.0)
        chain_mean[j] = np.mean(chains_matrix[:, j])
    W = np.mean(chain_var)
    if M > 1:
        B = np.var(chain_mean, ddof=1)
        var_plus = W * (N - 1.0) / N + B
    else:
        var_plus = W
    if var_plus <= 0:
        return float(N * M)
    rho_hat = []
    prev_pair = float('inf')
    max_lag = N - 3
    t = 0
    while t < max_lag:
        mean_ac = sum(acov_all[j][t] for j in range(M)) / M
        rho_t = 1.0 - (W - mean_ac) / var_plus
        if t + 1 >= N:
            rho_hat.append(rho_t)
            break
        mean_ac1 = sum(acov_all[j][t+1] for j in range(M)) / M
        rho_t1 = 1.0 - (W - mean_ac1) / var_plus
        pair = rho_t + rho_t1
        if pair < 0:
            break
        if pair > prev_pair:
            rho_t = prev_pair / 2.0
            rho_t1 = prev_pair / 2.0
            pair = prev_pair
        prev_pair = pair
        rho_hat.extend([rho_t, rho_t1])
        t += 2
    tau = -1.0 + 2.0 * sum(rho_hat)
    total = float(N * M)
    tau = max(tau, 1.0 / math.log10(total))
    return total / tau

def split_chains(chains):
    N, M = chains.shape
    nhalf = N // 2
    start2 = (N + 1) // 2
    split = np.zeros((nhalf, 2 * M))
    for j in range(M):
        split[:, 2*j] = chains[:nhalf, j]
        split[:, 2*j+1] = chains[start2:start2+nhalf, j]
    return split

def rank_transform(chains):
    N, M = chains.shape
    total = N * M
    pooled = chains.ravel(order='F')
    order = np.argsort(pooled, kind='stable')
    ranks = np.empty(total)
    k = 0
    while k < total:
        k_start = k
        while k + 1 < total and pooled[order[k+1]] == pooled[order[k]]:
            k += 1
        avg_rank = 0.5 * (k_start + k) + 1.0
        for t in range(k_start, k + 1):
            ranks[order[t]] = avg_rank
        k += 1
    denom = total + 0.25
    z = norm.ppf((ranks - 0.375) / denom)
    return z.reshape((N, M), order='F')

def compute_bulk_ess(chains):
    split = split_chains(chains)
    ranked = rank_transform(split)
    return ess_func(ranked)

# ── Run Stan at one H value ──

def run_stan(model, H, metric="dense_e"):
    data_file = DATA_DIR / f"data_H{H}.json"
    if not data_file.exists():
        print(f"  ERROR: {data_file} not found. Run run_phaseskate.jl first.")
        return None

    dim = 14 + 2 * H
    print(f"\n=== Stan: H={H}  dim={dim} ===")

    output_dir = SCRIPT_DIR / "stan_output"
    output_dir.mkdir(exist_ok=True)

    t0 = time.time()
    fit = model.sample(
        data=str(data_file),
        chains=NUM_CHAINS,
        parallel_chains=NUM_CHAINS,
        iter_sampling=NUM_SAMPLES,
        iter_warmup=NUM_WARMUP,
        max_treedepth=MAX_DEPTH,
        metric=metric,
        seed=42,
        show_console=False,
        output_dir=str(output_dir),
    )
    wall = time.time() - t0

    # Extract draws: (N, D, M) -> compute per-param bulk ESS
    draws = fit.draws()  # (N, M, D) in cmdstanpy
    n_draws, n_chains_out, n_params = draws.shape

    ess_bulk_vals = []
    for p in range(n_params):
        param_chains = draws[:, :, p]  # (N, M)
        ess_bulk_vals.append(compute_bulk_ess(param_chains))

    # Skip Stan's internal params (lp__, etc.) — they're first 7
    # cmdstanpy column_names tells us
    col_names = fit.column_names
    param_ess = []
    for i, name in enumerate(col_names):
        if not name.endswith("__"):
            param_ess.append(ess_bulk_vals[i])

    min_ess = min(param_ess)
    med_ess = float(np.median(param_ess))
    ess_per_s = min_ess / wall

    print(f"  Min ESS:    {min_ess:.1f}")
    print(f"  Median ESS: {med_ess:.1f}")
    print(f"  Wall time:  {wall:.1f} s")
    print(f"  ESS/s:      {ess_per_s:.1f}")

    return {"H": H, "dim": dim, "min_ess": min_ess,
            "median_ess": med_ess, "wall_time": wall, "ess_per_s": ess_per_s}


def main():
    import sys
    metric = sys.argv[1] if len(sys.argv) > 1 else "dense_e"
    suffix = "" if metric == "dense_e" else "_diagonal"

    print(f"Using CmdStan: {cmdstanpy.cmdstan_path()}")
    print(f"Metric: {metric}")

    # Compile once
    model = CmdStanModel(stan_file=str(STAN_MODEL))
    print(f"Compiled: {model.exe_file}")

    H_values = [10, 25, 50, 100]
    results = []
    for H in H_values:
        r = run_stan(model, H, metric=metric)
        if r:
            results.append(r)

    out_file = SCRIPT_DIR / f"results_stan{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
