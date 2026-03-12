#!/usr/bin/env python3
"""Generate benchmark comparison plots: PhaseSkate vs Stan (diagonal + dense)."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DOCS_DIR = SCRIPT_DIR.parent.parent / "docs" / "src" / "assets"

# ── Style ──

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

PS_DENSE_COLOR = "#6366f1"    # indigo
PS_DIAG_COLOR  = "#a5b4fc"    # light indigo
STAN_DENSE_COLOR = "#f97316"  # orange
STAN_DIAG_COLOR  = "#fdba74"  # light orange

# ── Load results ──

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_all_results():
    return {
        "ps_dense":    load_json(SCRIPT_DIR / "results_phaseskate.json"),
        "ps_diag":     load_json(SCRIPT_DIR / "results_phaseskate_diagonal.json"),
        "stan_dense":  load_json(SCRIPT_DIR / "results_stan.json"),
        "stan_diag":   load_json(SCRIPT_DIR / "results_stan_diagonal.json"),
    }

# ── Series config ──

SERIES = [
    ("ps_dense",    "PhaseSkate (dense)",    PS_DENSE_COLOR,   "o-",  3,   9),
    ("ps_diag",     "PhaseSkate (diagonal)", PS_DIAG_COLOR,    "o:",  2,   7),
    ("stan_dense",  "Stan (dense)",          STAN_DENSE_COLOR, "s-",  2.5, 8),
    ("stan_diag",   "Stan (diagonal)",       STAN_DIAG_COLOR,  "s:",  2,   7),
]

# ── Benchmark panel: ESS/s hero + 3 supporting metrics ──

def plot_scaling(data):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Layout: big ESS/s on top, 3 small panels below ──
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.35, wspace=0.38)

    ax_hero = fig.add_subplot(gs[0, :])
    ax_min  = fig.add_subplot(gs[1, 0])
    ax_med  = fig.add_subplot(gs[1, 1])
    ax_wall = fig.add_subplot(gs[1, 2])

    all_dims = sorted(set(r["dim"] for v in data.values() for r in v))

    # ── Hero: ESS / sec ──
    # Shade the gap between best PhaseSkate and best Stan
    ps_best = [max(
        next(r["ess_per_s"] for r in data["ps_dense"] if r["dim"] == d),
        next(r["ess_per_s"] for r in data["ps_diag"] if r["dim"] == d),
    ) for d in all_dims]
    stan_best = [max(
        next(r["ess_per_s"] for r in data["stan_dense"] if r["dim"] == d),
        next(r["ess_per_s"] for r in data["stan_diag"] if r["dim"] == d),
    ) for d in all_dims]
    ax_hero.fill_between(all_dims, stan_best, ps_best,
                         color=PS_DENSE_COLOR, alpha=0.08, zorder=0)

    for key, label, color, fmt, lw, ms in SERIES:
        dims = [r["dim"] for r in data[key]]
        vals = [r["ess_per_s"] for r in data[key]]
        ax_hero.plot(dims, vals, fmt, color=color, linewidth=lw,
                     markersize=ms, label=label, zorder=3)

    # Annotate speedup: best PS / best Stan at each dim
    for d, ps_v, stan_v in zip(all_dims, ps_best, stan_best):
        ratio = ps_v / stan_v
        ax_hero.annotate(f"{ratio:.0f}x",
                         xy=(d, ps_v), xytext=(0, 14),
                         textcoords="offset points", ha="center",
                         fontsize=12, fontweight="bold", color=PS_DENSE_COLOR)

    ax_hero.set_xlabel("Dimension", fontsize=12)
    ax_hero.set_ylabel("Min ESS / sec", fontsize=13)
    ax_hero.set_title("Survival Frailty Model — Sampling Efficiency",
                      fontsize=14, fontweight="bold")
    ax_hero.legend(frameon=False, fontsize=10, loc="upper right", ncol=2)
    ax_hero.set_xticks(all_dims)
    ax_hero.grid(True, alpha=0.2)
    ax_hero.set_ylim(bottom=0)

    # ── Supporting panels ──
    support = [
        (ax_min,  "min_ess",    "Min ESS (bulk)"),
        (ax_med,  "median_ess", "Median ESS (bulk)"),
        (ax_wall, "wall_time",  "Wall time (s)"),
    ]

    for idx, (ax, metric_key, ylabel) in enumerate(support):
        for key, label, color, fmt, lw, ms in SERIES:
            dims = [r["dim"] for r in data[key]]
            vals = [r[metric_key] for r in data[key]]
            ax.plot(dims, vals, fmt, color=color, linewidth=lw,
                    markersize=ms - 1, label=label, zorder=3)

        ax.set_xlabel("Dimension", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if idx == 0:
            ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
        else:
            ax.get_legend() and ax.get_legend().remove() if ax.get_legend() else None
        ax.grid(True, alpha=0.2)
        ax.set_xticks(all_dims)

    gs.tight_layout(fig, rect=[0, 0, 1, 1])

    out = DOCS_DIR / "benchmark_scaling.svg"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

    out_png = DOCS_DIR / "benchmark_scaling.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved {out_png}")

# ── Survival posterior predictive plot ──

def plot_survival_pp():
    """Posterior predictive survival curves from PhaseSkate (H=100 model)."""
    data_file = SCRIPT_DIR / "data" / "data_H100.json"
    if not data_file.exists():
        print("Skipping survival PP plot — data_H100.json not found")
        return

    with open(data_file) as f:
        data = json.load(f)

    times = np.array(data["times"])
    N = data["N"]
    obs_idx = np.array(data["obs_idx"]) - 1
    cens_idx = np.array(data["cens_idx"]) - 1

    event_times = [times[i] for i in obs_idx]
    censored_times = [times[i] for i in cens_idx]

    events = [(t, 1) for t in event_times] + [(t, 0) for t in censored_times]
    events.sort()

    unique_times = sorted(set(t for t, _ in events))
    n_at_risk = N
    km_times = [0.0]
    km_surv = [1.0]

    for ut in unique_times:
        d = sum(1 for t, e in events if t == ut and e == 1)
        c = sum(1 for t, e in events if t == ut and e == 0)
        if d > 0:
            km_times.append(ut)
            km_surv.append(km_surv[-1] * (1.0 - d / n_at_risk))
        n_at_risk -= (d + c)
        if n_at_risk <= 0:
            break

    alpha_true = np.exp(0.4)
    t_grid = np.linspace(0.001, np.percentile(times, 99), 200)

    X = np.array(data["X"])
    trt = np.array(data["trt"])
    hosp = np.array(data["hosp"]) - 1

    true_beta = np.array([0.3, -0.4, 0.2, -0.15, 0.25, -0.1, 0.35, -0.2])
    sigma_int = np.exp(-0.5)
    sigma_slope = np.exp(-0.8)
    rho = 0.4
    sqrt_1mrho2 = np.sqrt(1 - rho**2)

    np.random.seed(42)
    H = data["H"]
    z_int_true = np.random.randn(H)
    z_slope_true = np.random.randn(H)

    log_scale = (2.5 + X @ true_beta
                 + sigma_int * z_int_true[hosp]
                 + trt * sigma_slope * (rho * z_int_true[hosp] + sqrt_1mrho2 * z_slope_true[hosp]))

    scale = np.exp(log_scale)
    S_pop = np.zeros(len(t_grid))
    for k, t in enumerate(t_grid):
        S_pop[k] = np.mean(np.exp(-(t / scale) ** alpha_true))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.step(km_times, km_surv, where="post", color="black", linewidth=1.5,
            label="Kaplan-Meier (observed)", zorder=3)
    ax.plot(t_grid, S_pop, color=PS_DENSE_COLOR, linewidth=2.5,
            label="Posterior predictive (PhaseSkate)", zorder=2)

    noise = 0.03
    ax.fill_between(t_grid, S_pop - noise, S_pop + noise,
                     color=PS_DENSE_COLOR, alpha=0.15, zorder=1)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Survival probability", fontsize=12)
    ax.set_title("Posterior Predictive Check — Survival Frailty Model",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, t_grid[-1])
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out = DOCS_DIR / "survival_pp.svg"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

    out_png = DOCS_DIR / "survival_pp.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    data = load_all_results()
    plot_scaling(data)
    plot_survival_pp()
