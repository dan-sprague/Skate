#!/usr/bin/env julia
#
# Generate the landing-page 1×3 panel figure:
#   [A] Marginal survival by treatment arm  (from Stan posterior draws)
#   [B] Wall time / 1000 ESS vs dimension
#   [C] PhaseSkate vs Stan posterior means
#
# Run from docs/:  julia --project=. generate_example_plot.jl
#
using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using JSON3
import Statistics
using Statistics: mean, quantile
using Random
using LinearAlgebra: dot

const BENCH_DIR = joinpath(@__DIR__, "..", "benchmarks", "scaling")
const OUT_DIR   = joinpath(@__DIR__, "src", "assets")

# ── Theme (VitePress dark) ──────────────────────────────────────────────────
const BG    = RGBAf(0.106, 0.106, 0.122, 1.0)
const FG    = RGBAf(0.933, 0.933, 0.953, 1.0)
const GRID  = RGBAf(1, 1, 1, 0.08)
const MUTED = RGBAf(1, 1, 1, 0.4)
const SPINE = RGBAf(1, 1, 1, 0.25)

const PS_COLOR   = Makie.RGBf(0.251, 0.388, 0.847)   # #4063D8
const STAN_COLOR = Makie.RGBf(0.796, 0.235, 0.200)   # #CB3C33
const CTRL_COLOR = Makie.RGBf(0.584, 0.345, 0.698)   # Julia purple
const TRT_COLOR  = Makie.RGBf(0.259, 0.827, 0.573)   # #42d392 theme green

const SZ = 160   # panel size (pt)

load_json(path) = JSON3.read(read(path, String))

function style_axis!(ax)
    ax.backgroundcolor    = BG
    ax.xlabelcolor        = FG;  ax.ylabelcolor        = FG
    ax.titlecolor         = FG
    ax.xticklabelcolor    = FG;  ax.yticklabelcolor    = FG
    ax.xtickcolor         = SPINE; ax.ytickcolor       = SPINE
    ax.xgridcolor         = GRID;  ax.ygridcolor       = GRID
    ax.bottomspinecolor   = SPINE; ax.leftspinecolor   = SPINE
    ax.topspinevisible    = false; ax.rightspinevisible = false
end

# ── Load Stan posterior draws ───────────────────────────────────────────────
function load_stan_draws(H; n_subsample=150)
    dir = joinpath(BENCH_DIR, "stan_output")
    all_draws = Vector{Dict{String,Float64}}()
    header = String[]
    for c in 1:4
        f = joinpath(dir, "output_H$(H)_chain$(c).csv")
        isfile(f) || return nothing
        lines = readlines(f)
        data_lines = filter(l -> !startswith(l, '#'), lines)
        header = String.(split(data_lines[1], ','))
        for line in data_lines[2:end]
            vals = parse.(Float64, split(line, ','))
            d = Dict{String,Float64}()
            for (j, name) in enumerate(header)
                ps_name = replace(name, r"\.(\d+)" => s"[\1]")
                d[ps_name] = vals[j]
            end
            push!(all_draws, d)
        end
    end
    # Subsample
    Random.seed!(99)
    idx = sort(randperm(length(all_draws))[1:min(n_subsample, length(all_draws))])
    return all_draws[idx]
end

# ── Marginal survival by treatment arm for a single draw ────────────────────
function marginal_survival(ppdata, draw, t_grid)
    N    = ppdata.N
    X    = reduce(hcat, ppdata.X)' |> collect
    trt  = Float64.(ppdata.trt)
    hosp = Int.(ppdata.hosp)
    H    = ppdata.H

    alpha       = exp(draw["log_alpha"])
    beta_0      = draw["beta_0"]
    beta        = [draw["beta[$i]"] for i in 1:8]
    sigma_int   = exp(draw["log_sigma_int"])
    sigma_slope = exp(draw["log_sigma_slope"])
    rho         = draw["rho"]
    z_int       = [draw["z_int[$i]"] for i in 1:H]
    z_slope     = [draw["z_slope[$i]"] for i in 1:H]
    sqrt_1mrho2 = sqrt(max(1 - rho^2, 0.0))

    log_scale = Vector{Float64}(undef, N)
    trt_eff     = draw["trt_effect"]
    for i in 1:N
        log_scale[i] = beta_0 + dot(@view(X[i,:]), beta) +
            sigma_int * z_int[hosp[i]] +
            trt[i] * (trt_eff + sigma_slope * (rho * z_int[hosp[i]] + sqrt_1mrho2 * z_slope[hosp[i]]))
    end
    scale = exp.(log_scale)

    ctrl_idx = findall(==(0.0), trt)
    trt_idx  = findall(==(1.0), trt)

    nt = length(t_grid)
    S_ctrl = Vector{Float64}(undef, nt)
    S_trt  = Vector{Float64}(undef, nt)
    for (k, t) in enumerate(t_grid)
        S_ctrl[k] = mean(exp.(-(t ./ @view(scale[ctrl_idx])) .^ alpha))
        S_trt[k]  = mean(exp.(-(t ./ @view(scale[trt_idx]))  .^ alpha))
    end
    return S_ctrl, S_trt
end

# ══════════════════════════════════════════════════════════════════════════════
function build_figure()

# ── Data ────────────────────────────────────────────────────────────────────
ppdata = load_json(joinpath(BENCH_DIR, "data", "data_H100.json"))
draws  = load_stan_draws(100; n_subsample=150)
times  = Float64.(ppdata.times)
trt    = Float64.(ppdata.trt)

t_grid = collect(range(0.001, 25.0; length=150))
nt     = length(t_grid)
nd     = length(draws)

# Compute marginal survival for each draw
S_ctrl_mat = Matrix{Float64}(undef, nt, nd)
S_trt_mat  = Matrix{Float64}(undef, nt, nd)
for (j, d) in enumerate(draws)
    S_ctrl_mat[:, j], S_trt_mat[:, j] = marginal_survival(ppdata, d, t_grid)
end

# Quantile bands
qband(mat, lo, hi) = ([quantile(mat[i,:], lo) for i in 1:size(mat,1)],
                       [quantile(mat[i,:], hi) for i in 1:size(mat,1)])
S_ctrl_mean = vec(mean(S_ctrl_mat; dims=2))
S_trt_mean  = vec(mean(S_trt_mat;  dims=2))
ctrl_95 = qband(S_ctrl_mat, 0.025, 0.975)
ctrl_80 = qband(S_ctrl_mat, 0.10,  0.90)
trt_95  = qband(S_trt_mat,  0.025, 0.975)
trt_80  = qband(S_trt_mat,  0.10,  0.90)

# Kaplan-Meier by arm
function kaplan_meier(times, observed, mask)
    idx = findall(mask)
    t_sub = times[idx]; obs_sub = observed[idx]
    order = sortperm(t_sub)
    km_t = [0.0]; km_s = [1.0]; at_risk = length(idx); surv = 1.0
    for i in order
        if obs_sub[i]
            surv *= (at_risk - 1) / at_risk
            push!(km_t, t_sub[i]); push!(km_s, surv)
        end
        at_risk -= 1
    end
    return km_t, km_s
end

obs_idx  = Int.(ppdata.obs_idx)
observed = falses(ppdata.N); observed[obs_idx] .= true
km_ctrl_t, km_ctrl_s = kaplan_meier(times, observed, trt .== 0.0)
km_trt_t,  km_trt_s  = kaplan_meier(times, observed, trt .== 1.0)

# ── Compute acceleration factor from trt_effect ───────────────────────────
# Labels are swapped (control ↔ treatment), so negate trt_effect.
# AF > 1 means treatment (the higher curve) prolongs survival.
af_draws = Vector{Float64}(undef, nd)
for (j, d) in enumerate(draws)
    af_draws[j] = exp(-d["trt_effect"])
end
af_mean  = mean(af_draws)
af_std   = Statistics.std(af_draws)
p_af_gt1 = mean(af_draws .> 1.0)

# ── Figure (1×3 row) ──────────────────────────────────────────────────────
fig = Figure(; backgroundcolor=BG, fontsize=11,
    figure_padding=(12, 16, 8, 12))

# ── [A] Marginal survival by treatment arm ── [1,1] ─────────────────────────
ax_a = Axis(fig[1, 3]; width=SZ, height=SZ,
    xlabel="Time (sim. days)", ylabel="Marginal Survival Probability",
    xlabelsize=10, ylabelsize=10, xticklabelsize=9, yticklabelsize=9,
)
style_axis!(ax_a)

# 95% CI bands + dotted edge lines for visibility
band!(ax_a, t_grid, ctrl_95[1], ctrl_95[2]; color=(CTRL_COLOR, 0.15))
lines!(ax_a, t_grid, ctrl_95[1]; color=(CTRL_COLOR, 0.4), linewidth=0.7, linestyle=:dot)
lines!(ax_a, t_grid, ctrl_95[2]; color=(CTRL_COLOR, 0.4), linewidth=0.7, linestyle=:dot)
band!(ax_a, t_grid, trt_95[1],  trt_95[2];  color=(TRT_COLOR,  0.15))
lines!(ax_a, t_grid, trt_95[1];  color=(TRT_COLOR, 0.4), linewidth=0.7, linestyle=:dot)
lines!(ax_a, t_grid, trt_95[2];  color=(TRT_COLOR, 0.4), linewidth=0.7, linestyle=:dot)

# KM stairs
stairs!(ax_a, km_ctrl_t, km_ctrl_s; color=(CTRL_COLOR, 0.45), linewidth=1.2, step=:post)
stairs!(ax_a, km_trt_t,  km_trt_s;  color=(TRT_COLOR,  0.45), linewidth=1.2, step=:post)

# Posterior mean curves
lines!(ax_a, t_grid, S_ctrl_mean; color=CTRL_COLOR, linewidth=2.5, label="Control")
lines!(ax_a, t_grid, S_trt_mean;  color=TRT_COLOR,  linewidth=2.5, label="Treatment")

ylims!(ax_a, -0.02, 1.02); xlims!(ax_a, 0, t_grid[end])

p_af_str = p_af_gt1 == 0.0 ? "< $(round(1.0/nd; digits=2))" : p_af_gt1 == 1.0 ? "> $(round(1.0 - 1.0/nd; digits=2))" : "$(round(p_af_gt1; digits=3))"
af_text = "STR = $(round(af_mean; digits=2)) ± $(round(af_std; digits=2))\nP(STR > 1) $p_af_str"
text!(ax_a, 0.97, 0.60; space=:relative, align=(:right, :top), fontsize=8, color=FG,
    text=af_text)

Legend(fig[1, 3], ax_a;
    halign=:right, valign=:top, margin=(0, 10, 10, 0),
    framevisible=false, labelcolor=FG, labelsize=9,
    backgroundcolor=RGBAf(0.1, 0.1, 0.12, 0.75), padding=(8, 8, 4, 4))

# ── [B] Wall time / 1000 ESS ── [1,2] ──────────────────────────────────────
ps_dense = load_json(joinpath(BENCH_DIR, "results_phaseskate.json"))
ps_diag  = load_json(joinpath(BENCH_DIR, "results_phaseskate_diagonal.json"))
st_dense = load_json(joinpath(BENCH_DIR, "results_stan.json"))
st_diag  = load_json(joinpath(BENCH_DIR, "results_stan_diagonal.json"))

get_field(data, f) = [r[f] for r in data]

ax_b = Axis(fig[1, 1]; width=SZ, height=SZ,
    xlabel="Dimension", ylabel="Wall time (s) / 1000 ESS",
    xlabelsize=10, ylabelsize=10, xticklabelsize=9, yticklabelsize=9,
)
style_axis!(ax_b)

all_dims = sort(unique(Int.(get_field(ps_dense, :dim))))
ax_b.xticks = (Float64.(all_dims), string.(all_dims))

for (data, label, color, lstyle, lw) in [
    (ps_dense,  "PhaseSkate (dense)",    PS_COLOR,   :solid, 2.5),
    (ps_diag,   "PhaseSkate (diagonal)", PS_COLOR,   :dash,  1.8),
    (st_dense,  "Stan (dense)",          STAN_COLOR, :solid, 2.5),
    (st_diag,   "Stan (diagonal)",       STAN_COLOR, :dash,  1.8),
]
    dims = Float64.(get_field(data, :dim))
    wall = Float64.(get_field(data, :wall_time))
    ess  = Float64.(get_field(data, :min_ess))
    vals = (wall ./ ess) .* 1000
    scatterlines!(ax_b, dims, vals;
        color=color, linewidth=lw, linestyle=lstyle,
        marker=:circle, markersize=6, label=label)
end
xlims!(ax_b, minimum(all_dims) - 5, maximum(all_dims) + 15)

# ── [C] PS vs Stan posterior means ── [1,3] ─────────────────────────────────
pm = load_json(joinpath(BENCH_DIR, "posterior_means.json"))
stan_m = Float64[p.stan_mean for p in pm]
ps_m   = Float64[p.ps_mean   for p in pm]

ax_c = Axis(fig[1, 2]; width=SZ, height=SZ,
    xlabel="Stan posterior mean", ylabel="PhaseSkate posterior mean",
    xlabelsize=10, ylabelsize=10, xticklabelsize=9, yticklabelsize=9,
)
style_axis!(ax_c)

lo = min(minimum(stan_m), minimum(ps_m))
hi = max(maximum(stan_m), maximum(ps_m))
pad = 0.05 * (hi - lo)
lines!(ax_c, [lo-pad, hi+pad], [lo-pad, hi+pad]; color=MUTED, linewidth=1.5, linestyle=:dash)
scatter!(ax_c, stan_m, ps_m; color=(FG, 0.5), markersize=4, marker=:circle)

ss_res  = sum((ps_m .- stan_m).^2)
ss_tot  = sum((ps_m .- mean(ps_m)).^2)
r2      = 1.0 - ss_res / ss_tot
max_dev = maximum(abs.(ps_m .- stan_m))

text!(ax_c, 0.05, 0.95; space=:relative, align=(:left, :top), fontsize=8, color=FG,
    text="R² = $(round(r2; digits=6))\nmax|Δ| = $(round(max_dev; sigdigits=2))\nd = 214")

# ── Panel labels ────────────────────────────────────────────────────────────
for ((row, col), lbl) in [((1,1), "A"), ((1,2), "B"), ((1,3), "C")]
    # A=scaling, B=posterior means, C=survival
    Label(fig[row, col, TopLeft()], lbl; fontsize=16, color=FG, font=:bold, padding=(0,0,0,0))
end

# ── Legend inside panel B ─────────────────────────────────────────────────
Legend(fig[1, 1], ax_b;
    halign=:left, valign=:top, margin=(8, 0, 8, 0),
    framevisible=false, labelcolor=FG, labelsize=8,
    backgroundcolor=RGBAf(0.1, 0.1, 0.12, 0.75), padding=(6, 6, 4, 4),
    patchsize=(10, 6),
)

# ── Spacing ─────────────────────────────────────────────────────────────────
colgap!(fig.layout, 14)
resize_to_layout!(fig)

# ── Save ────────────────────────────────────────────────────────────────────
mkpath(OUT_DIR)
outpath = joinpath(OUT_DIR, "survival_example.svg")
save(outpath, fig)
println("Saved → $outpath")
return fig
end

build_figure()
