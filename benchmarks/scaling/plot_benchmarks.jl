# benchmarks/scaling/plot_benchmarks.jl
# Generate benchmark comparison plots using CairoMakie.
# Produces: benchmark_scaling.svg/png
#
# Prerequisites: ] add CairoMakie JSON3
# Usage: julia benchmarks/scaling/plot_benchmarks.jl

using CairoMakie
using JSON3
using Statistics
using Random
using LinearAlgebra: dot

const SCRIPT_DIR = @__DIR__
const DOCS_DIR = joinpath(SCRIPT_DIR, "..", "..", "docs", "src", "assets")

# ── Julia logo colors ──

const JULIA_BLUE   = Makie.RGBf(0.251, 0.388, 0.847)  # #4063D8
const JULIA_GREEN  = Makie.RGBf(0.220, 0.596, 0.149)  # #389826
const JULIA_RED    = Makie.RGBf(0.796, 0.235, 0.200)  # #CB3C33
const JULIA_PURPLE = Makie.RGBf(0.584, 0.345, 0.698)  # #9558B2

const PS_COLOR   = JULIA_BLUE
const STAN_COLOR = JULIA_RED

# ── Load results ──

load_json(path) = JSON3.read(read(path, String))

function load_all()
    Dict(
        :ps_dense   => load_json(joinpath(SCRIPT_DIR, "results_phaseskate.json")),
        :ps_diag    => load_json(joinpath(SCRIPT_DIR, "results_phaseskate_diagonal.json")),
        :stan_dense => load_json(joinpath(SCRIPT_DIR, "results_stan.json")),
        :stan_diag  => load_json(joinpath(SCRIPT_DIR, "results_stan_diagonal.json")),
    )
end

function load_posterior_comparison()
    path = joinpath(SCRIPT_DIR, "posterior_comparison.json")
    isfile(path) || return nothing
    return JSON3.read(read(path, String))
end

function load_posterior_means()
    path = joinpath(SCRIPT_DIR, "posterior_means.json")
    isfile(path) || return nothing
    return JSON3.read(read(path, String))
end

function load_stan_posterior_means(H)
    dir = joinpath(SCRIPT_DIR, "stan_output")
    chains = Matrix{Float64}[]
    header = String[]
    for c in 1:4
        f = joinpath(dir, "output_H$(H)_chain$(c).csv")
        isfile(f) || return nothing
        lines = readlines(f)
        data_lines = filter(l -> !startswith(l, '#'), lines)
        header = split(data_lines[1], ',')
        nrows = length(data_lines) - 1
        ncols = length(header)
        mat = Matrix{Float64}(undef, nrows, ncols)
        for (i, line) in enumerate(data_lines[2:end])
            mat[i, :] .= parse.(Float64, split(line, ','))
        end
        push!(chains, mat)
    end
    pooled = vcat(chains...)
    means = Dict{String, Float64}()
    for (j, name) in enumerate(header)
        endswith(name, "__") && continue
        ps_name = replace(string(name), r"\.(\d+)" => s"[\1]")
        means[ps_name] = Statistics.mean(pooled[:, j])
    end
    return means
end

# ── Theme ──

function make_theme(mode::Symbol)
    if mode == :dark
        bg = RGBAf(0.106, 0.106, 0.122, 1.0)
        fg = RGBAf(0.933, 0.933, 0.953, 1.0)
        grid = RGBAf(1, 1, 1, 0.08)
        muted = RGBAf(1, 1, 1, 0.4)
    else
        bg = RGBAf(1, 1, 1, 1.0)
        fg = RGBAf(0.15, 0.15, 0.18, 1.0)
        grid = RGBAf(0, 0, 0, 0.06)
        muted = RGBAf(0, 0, 0, 0.4)
    end
    return (; bg, fg, grid, muted)
end

function style_axis!(ax, th)
    ax.backgroundcolor = th.bg
    ax.xlabelcolor = th.fg
    ax.ylabelcolor = th.fg
    ax.xticklabelcolor = th.fg
    ax.yticklabelcolor = th.fg
    ax.xtickcolor = th.muted
    ax.ytickcolor = th.muted
    ax.xgridcolor = th.grid
    ax.ygridcolor = th.grid
    ax.bottomspinecolor = th.muted
    ax.leftspinecolor = th.muted
    ax.topspinevisible = false
    ax.rightspinevisible = false
end

# ── Series config ──

const MS = 8  # uniform markersize

const SERIES = [
    (:ps_dense,   "PhaseSkate (dense)",    PS_COLOR,   :circle, :solid,  2.5),
    (:ps_diag,    "PhaseSkate (diagonal)", PS_COLOR,   :circle, :dash,   2),
    (:stan_dense, "Stan (dense)",          STAN_COLOR, :circle, :solid,  2.5),
    (:stan_diag,  "Stan (diagonal)",       STAN_COLOR, :circle, :dash,   2),
]

function get_vals(data, key, field)
    [r[field] for r in data[key]]
end
get_dims(data, key) = get_vals(data, key, :dim)

# ── Compute survival curve from parameter values ──

function compute_survival_curve(ppdata, alpha, beta_0, beta, sigma_int, sigma_slope,
                                 rho, z_int, z_slope)
    times = Float64.(ppdata.times)
    N = ppdata.N
    X = reduce(hcat, ppdata.X)' |> collect
    trt = Float64.(ppdata.trt)
    hosp = Int.(ppdata.hosp)
    sqrt_1mrho2 = sqrt(1 - rho^2)

    t_grid = range(0.001, quantile(times, 0.99), length=200) |> collect

    log_scale = [beta_0 + dot(X[i,:], beta) +
                 sigma_int * z_int[hosp[i]] +
                 trt[i] * sigma_slope * (rho * z_int[hosp[i]] + sqrt_1mrho2 * z_slope[hosp[i]])
                 for i in 1:N]
    scale = exp.(log_scale)
    S_pop = [mean(exp.(-(t ./ scale) .^ alpha)) for t in t_grid]
    return t_grid, S_pop
end

# ── Main plot ──

const SZ = 175  # panel size

# Register Inter font from local files
const _INTER_DIR = joinpath(homedir(), "Downloads", "Inter", "static")
const FONT      = CairoMakie.to_font(joinpath(_INTER_DIR, "Inter_18pt-Regular.ttf"))
const FONT_BOLD = CairoMakie.to_font(joinpath(_INTER_DIR, "Inter_18pt-Bold.ttf"))

function plot_main(data; mode=:light)
    th = make_theme(mode)
    all_dims = sort(unique(vcat([get_dims(data, k) for (k,_,_,_,_,_) in SERIES]...)))
    pc = load_posterior_comparison()
    pm = load_posterior_means()

    # Layout: row 0 = title, row 1 = [A][B][C], row 2 = legends
    fig = Figure(backgroundcolor=th.bg, fontsize=11, font=FONT)

    # ════════════════════════════════════════════
    # [A] Survival Posterior Predictive — row 1, col 1
    # ════════════════════════════════════════════
    data_file = joinpath(SCRIPT_DIR, "data", "data_H100.json")
    ax_pp = Axis(fig[1, 1];
        width = SZ, height = SZ,
        xlabel = "Time (sim. days)", ylabel = "S(t)",
        xlabelsize = 10, ylabelsize = 10,
        xlabelfont = FONT, ylabelfont = FONT,
        xticklabelsize = 9, yticklabelsize = 9,
    )
    style_axis!(ax_pp, th)

    if isfile(data_file)
        ppdata = JSON3.read(read(data_file, String))
        times = Float64.(ppdata.times)
        N = ppdata.N
        obs_idx = Int.(ppdata.obs_idx)
        observed = falses(N)
        observed[obs_idx] .= true

        order = sortperm(times)
        km_t = [0.0]; km_s = [1.0]; at_risk = N; surv = 1.0
        for i in order
            if observed[i]
                surv *= (at_risk - 1) / at_risk
                push!(km_t, times[i]); push!(km_s, surv)
            end
            at_risk -= 1
        end

        true_beta = [0.3, -0.4, 0.2, -0.15, 0.25, -0.1, 0.35, -0.2]
        Random.seed!(42)
        H = ppdata.H
        z_int_true = randn(H); z_slope_true = randn(H)

        t_grid, S_ps = compute_survival_curve(ppdata, exp(0.4), 2.5, true_beta,
            exp(-0.5), exp(-0.8), 0.4, z_int_true, z_slope_true)

        stan_means = load_stan_posterior_means(100)
        if stan_means !== nothing
            alpha_stan = exp(get(stan_means, "log_alpha", 0.4))
            beta_0_stan = get(stan_means, "beta_0", 2.5)
            beta_stan = [get(stan_means, "beta[$i]", true_beta[i]) for i in 1:8]
            sigma_int_stan = exp(get(stan_means, "log_sigma_int", -0.5))
            sigma_slope_stan = exp(get(stan_means, "log_sigma_slope", -0.8))
            rho_stan = get(stan_means, "rho", 0.4)
            z_int_stan = [get(stan_means, "z_int[$i]", 0.0) for i in 1:H]
            z_slope_stan = [get(stan_means, "z_slope[$i]", 0.0) for i in 1:H]

            _, S_stan = compute_survival_curve(ppdata, alpha_stan, beta_0_stan, beta_stan,
                sigma_int_stan, sigma_slope_stan, rho_stan, z_int_stan, z_slope_stan)

            band!(ax_pp, t_grid, S_stan .- 0.03, S_stan .+ 0.03;
                color = (STAN_COLOR, 0.10))
            lines!(ax_pp, t_grid, S_stan;
                color = STAN_COLOR, linewidth = 2, linestyle = :solid,
                label = "Stan")
        end

        band!(ax_pp, t_grid, S_ps .- 0.03, S_ps .+ 0.03;
            color = (PS_COLOR, 0.15))
        stairs!(ax_pp, km_t, km_s;
            color = th.fg, linewidth = 1.5, step = :post,
            label = "Kaplan-Meier")
        lines!(ax_pp, t_grid, S_ps;
            color = PS_COLOR, linewidth = 2,
            label = "PhaseSkate")

        ylims!(ax_pp, -0.02, 1.02)
        xlims!(ax_pp, 0, t_grid[end])

        text!(ax_pp, 0.97, 0.65;
            text = "S(t) ~ Weibull(α, Xβ)\nCorrelated frailty\nH = 100 sites, d = 213",
            space = :relative,
            align = (:right, :top),
            fontsize = 8,
            font = FONT,
            color = th.muted)
    end

    # ════════════════════════════════════════════
    # [B] Wall time per 1000 ESS — row 1, col 2
    # ════════════════════════════════════════════
    ax_cost = Axis(fig[1, 2];
        width = SZ, height = SZ,
        xlabel = "Dimension", ylabel = "Wall time (s) / 1000 ESS",
        xlabelsize = 10, ylabelsize = 10,
        xlabelfont = FONT, ylabelfont = FONT,
        xticklabelsize = 9, yticklabelsize = 9,
    )
    style_axis!(ax_cost, th)
    ax_cost.xticks = (all_dims, string.(all_dims))

    for (key, label, color, marker, lstyle, lw) in SERIES
        dims = get_dims(data, key)
        wall = get_vals(data, key, :wall_time)
        ess = get_vals(data, key, :min_ess)
        vals = (wall ./ ess) .* 1000
        scatterlines!(ax_cost, Float64.(dims), vals;
            color = color, linewidth = lw, linestyle = lstyle,
            marker = marker, markersize = MS,
            label = label)
    end

    xlims!(ax_cost, minimum(all_dims) - 5, maximum(all_dims) + 15)

    # ════════════════════════════════════════════
    # [C] Scatter plot — PS vs Stan posterior means — row 1, col 3
    # ════════════════════════════════════════════
    if pm !== nothing
        stan_means_c = Float64[p.stan_mean for p in pm]
        ps_means_c   = Float64[p.ps_mean for p in pm]

        ax_scatter = Axis(fig[1, 3];
            width = SZ, height = SZ,
            xlabel = "Stan posterior mean", ylabel = "PhaseSkate posterior mean",
            xlabelsize = 10, ylabelsize = 10,
            xlabelfont = FONT, ylabelfont = FONT,
            xticklabelsize = 9, yticklabelsize = 9,
        )
        style_axis!(ax_scatter, th)

        # y = x reference line
        lo = min(minimum(stan_means_c), minimum(ps_means_c))
        hi = max(maximum(stan_means_c), maximum(ps_means_c))
        pad = 0.05 * (hi - lo)
        lines!(ax_scatter, [lo - pad, hi + pad], [lo - pad, hi + pad];
            color = th.muted, linewidth = 1.5, linestyle = :dash)

        scatter!(ax_scatter, stan_means_c, ps_means_c;
            color = (th.fg, 0.5), markersize = 4, marker = :circle)

        # R² annotation
        ss_res = sum((ps_means_c .- stan_means_c).^2)
        ss_tot = sum((ps_means_c .- mean(ps_means_c)).^2)
        r2 = 1.0 - ss_res / ss_tot
        max_dev = maximum(abs.(ps_means_c .- stan_means_c))

        text!(ax_scatter, 0.05, 0.95;
            text = "R² = $(round(r2; digits=6))\nmax|Δ| = $(round(max_dev; sigdigits=2))\nd = 213",
            space = :relative,
            align = (:left, :top),
            fontsize = 8,
            font = FONT,
            color = th.fg)
    end

    # ════════════════════════════════════════════
    # Legends — row 2 (bottom)
    # ════════════════════════════════════════════

    # Sampler legend — row 2, cols 1:2
    Legend(fig[2, 1:2],
        [
            LineElement(color = PS_COLOR, linewidth = 2.5, linestyle = :solid),
            LineElement(color = STAN_COLOR, linewidth = 2.5, linestyle = :solid),
            LineElement(color = :gray50, linewidth = 2, linestyle = :solid),
            LineElement(color = :gray50, linewidth = 2, linestyle = :dash),
        ],
        [
            "PhaseSkate",
            "Stan",
            "Dense",
            "Diagonal",
        ];
        backgroundcolor = :transparent,
        framevisible = false,
        labelcolor = th.fg,
        labelfont = FONT,
        labelsize = 10,
        orientation = :horizontal,
        nbanks = 1,
        patchsize = (18, 12),
    )

    # Scatter legend — row 2, col 3
    if pm !== nothing
        Legend(fig[2, 3],
            [
                MarkerElement(marker = :circle, color = (th.fg, 0.5), markersize = 4),
                LineElement(color = th.muted, linewidth = 1.5, linestyle = :dash),
            ],
            [
                "Parameter (n=213)",
                "y = x",
            ];
            backgroundcolor = :transparent,
            framevisible = false,
            labelcolor = th.fg,
            labelfont = FONT,
            labelsize = 10,
            orientation = :horizontal,
            nbanks = 1,
            patchsize = (18, 12),
        )
    end

    # ── Panel labels ──
    Label(fig[1, 1, TopLeft()], "A";
        fontsize = 18, font = FONT_BOLD, color = th.fg, padding = (0, 0, 0, 0))
    Label(fig[1, 2, TopLeft()], "B";
        fontsize = 18, font = FONT_BOLD, color = th.fg, padding = (0, 0, 0, 0))
    Label(fig[1, 3, TopLeft()], "C";
        fontsize = 18, font = FONT_BOLD, color = th.fg, padding = (0, 0, 0, 0))

    # ── Supertitle — left-justified ──
    Label(fig[0, 0:3], "Enzyme LLVM autodiff enables fast sampling of complex posteriors";
        fontsize = 18,
        font = FONT_BOLD,
        color = th.fg,
        halign = :left,
        valign = :bottom,
        padding = (0, 0, 0, 6),
    )

    resize_to_layout!(fig)

    mkpath(DOCS_DIR)
    for ext in ["svg", "png"]
        out = joinpath(DOCS_DIR, "benchmark_scaling.$(ext)")
        save(out, fig; px_per_unit = ext == "png" ? 3 : 1)
        println("Saved $out")
    end

    return fig
end

# ── Main ──

data = load_all()
plot_main(data; mode=:light)
println("\nDone.")
