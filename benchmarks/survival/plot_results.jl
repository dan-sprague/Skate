# benchmarks/survival/plot_results.jl
# Generate ESS/s comparison bar chart and survival curve plot.
# Produces both light and dark theme versions for the docs site.
#
# Prerequisites: ] add CairoMakie JSON
# Usage: julia benchmarks/survival/plot_results.jl

using CairoMakie
using Random
using Distributions: Weibull

# ── Minimal JSON reader ──────────────────────────────────────────────────────

function read_json(path)
    txt = read(path, String)
    return _parse_json(txt, Ref(1))
end

function _skip_ws(s, pos)
    while pos[] <= length(s) && s[pos[]] in (' ', '\t', '\n', '\r')
        pos[] += 1
    end
end

function _parse_json(s, pos)
    _skip_ws(s, pos); c = s[pos[]]
    if c == '{';     return _parse_obj(s, pos)
    elseif c == '['; return _parse_arr(s, pos)
    elseif c == '"'; return _parse_str(s, pos)
    elseif c == 't'; pos[] += 4; return true
    elseif c == 'f'; pos[] += 5; return false
    elseif c == 'n'; pos[] += 4; return nothing
    else;            return _parse_num(s, pos)
    end
end

function _parse_str(s, pos)
    pos[] += 1; start = pos[]
    while s[pos[]] != '"'; s[pos[]] == '\\' && (pos[] += 1); pos[] += 1; end
    result = s[start:pos[]-1]; pos[] += 1; return result
end

function _parse_num(s, pos)
    start = pos[]
    while pos[] <= length(s) && s[pos[]] in ('-', '+', '.', 'e', 'E', '0':'9'...); pos[] += 1; end
    numstr = s[start:pos[]-1]
    (occursin('.', numstr) || occursin('e', numstr) || occursin('E', numstr)) ?
        parse(Float64, numstr) : parse(Int, numstr)
end

function _parse_arr(s, pos)
    pos[] += 1; result = Any[]
    _skip_ws(s, pos); s[pos[]] == ']' && (pos[] += 1; return result)
    while true
        push!(result, _parse_json(s, pos)); _skip_ws(s, pos)
        s[pos[]] == ',' ? (pos[] += 1) : break
    end
    pos[] += 1; return result
end

function _parse_obj(s, pos)
    pos[] += 1; result = Dict{String, Any}()
    _skip_ws(s, pos); s[pos[]] == '}' && (pos[] += 1; return result)
    while true
        _skip_ws(s, pos); key = _parse_str(s, pos)
        _skip_ws(s, pos); pos[] += 1; result[key] = _parse_json(s, pos)
        _skip_ws(s, pos); s[pos[]] == ',' ? (pos[] += 1) : break
    end
    pos[] += 1; return result
end

# ── Configuration ────────────────────────────────────────────────────────────

bench_dir  = @__DIR__
result_dir = joinpath(bench_dir, "results")
data_dir   = joinpath(bench_dir, "data")
plot_dir   = joinpath(bench_dir, "plots")
mkpath(plot_dir)

# Framework colors (vibrant, readable on both dark and light backgrounds)
const COLORS = Dict(
    "PhaseSkate" => "#4C9EEB",   # bright blue
    "CmdStan"    => "#E05252",   # warm red
    "Turing.jl"  => "#A86ED7",   # purple
)

const FRAMEWORK_ORDER = ["CmdStan", "Turing.jl", "PhaseSkate"]

# ── Load results ─────────────────────────────────────────────────────────────

function load_results()
    results = Dict{String, Any}()
    for f in ["phaseskate_results.json", "stan_results.json", "turing_results.json"]
        path = joinpath(result_dir, f)
        if isfile(path)
            r = read_json(path)
            results[r["backend"]] = r
        end
    end
    return results
end

# ── Kaplan-Meier estimator ───────────────────────────────────────────────────

function kaplan_meier(times, observed)
    n = length(times)
    order = sortperm(times)
    sorted_t = times[order]
    sorted_obs = observed[order]

    km_t = [0.0]
    km_s = [1.0]
    at_risk = n
    surv = 1.0

    for i in 1:n
        if sorted_obs[i]
            surv *= (at_risk - 1) / at_risk
            push!(km_t, sorted_t[i])
            push!(km_s, surv)
        end
        at_risk -= 1
    end
    return km_t, km_s
end

# ── Theme helpers ────────────────────────────────────────────────────────────

function make_theme(mode::Symbol)
    if mode == :dark
        bg = RGBAf(0.106, 0.106, 0.122, 1.0)     # #1b1b1f (VitePress dark)
        fg = RGBAf(0.933, 0.933, 0.953, 1.0)       # #eeeeF3
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

# ── Plot 1: ESS/s Bar Chart ─────────────────────────────────────────────────

function plot_ess_bars(results; mode=:dark)
    th = make_theme(mode)

    names = String[]
    vals  = Float64[]
    colors = RGBAf[]

    for fw in FRAMEWORK_ORDER
        haskey(results, fw) || continue
        push!(names, fw)
        push!(vals, results[fw]["ess_per_sec"])
        push!(colors, parse(Colorant, COLORS[fw]))
    end

    fig = Figure(size=(700, 300), backgroundcolor=th.bg)
    ax = Axis(fig[1, 1];
        xlabel = "Min ESS / sec",
        backgroundcolor = th.bg,
        xlabelcolor = th.fg,
        ylabelcolor = th.fg,
        xticklabelcolor = th.fg,
        yticklabelcolor = th.fg,
        xtickcolor = th.muted,
        ytickcolor = :transparent,
        xgridcolor = th.grid,
        ygridvisible = false,
        bottomspinecolor = th.muted,
        topspinevisible = false,
        rightspinevisible = false,
        leftspinevisible = false,
        xlabelsize = 16,
        xticklabelsize = 14,
        yticklabelsize = 15,
        xlabelfont = :bold,
    )

    positions = 1:length(names)
    barplot!(ax, positions, vals;
        direction = :x,
        color = colors,
        bar_labels = :values,
        label_formatter = x -> string(round(Int, x)),
        label_color = th.fg,
        label_size = 14,
        label_offset = 8,
        gap = 0.3,
        width = 0.6,
    )

    ax.yticks = (positions, names)
    xlims!(ax, 0, maximum(vals) * 1.25)

    save(joinpath(plot_dir, "ess_per_sec_$(mode).svg"), fig)
    save(joinpath(plot_dir, "ess_per_sec_$(mode).png"), fig, px_per_unit=3)
    println("Saved ESS/s plot ($mode)")
    return fig
end

# ── Plot 2: Survival Curves ─────────────────────────────────────────────────

function plot_survival_curves(; mode=:dark)
    th = make_theme(mode)

    # Load data and true params
    data = read_json(joinpath(data_dir, "survival_data.json"))
    true_p = read_json(joinpath(data_dir, "true_params.json"))

    times = Float64.(data["times"])
    trt   = Float64.(data["trt"])
    hosp  = Int.(data["hosp"])
    obs_idx  = Int.(data["obs_idx"])
    cens_idx = Int.(data["cens_idx"])
    observed = falses(length(times))
    observed[obs_idx] .= true

    alpha  = exp(true_p["log_alpha"])
    beta_0 = true_p["beta_0"]
    frailty_int   = Float64.(true_p["frailty_int"])
    frailty_slope = Float64.(true_p["frailty_slope"])

    # Masks
    ctrl_mask = trt .== 0.0
    trt_mask  = trt .== 1.0

    # KM curves
    km_ctrl_t, km_ctrl_s = kaplan_meier(times[ctrl_mask], observed[ctrl_mask])
    km_trt_t,  km_trt_s  = kaplan_meier(times[trt_mask],  observed[trt_mask])

    # True survival curves (average covariates = 0, median hospital frailty)
    t_grid = range(0, quantile(times, 0.98), length=200)

    # Median hospital frailties
    med_fint = median(frailty_int)
    med_fslope = median(frailty_slope)

    # S(t) = exp(-(t/scale)^alpha) where log_scale = beta_0 + frailty
    log_scale_ctrl = beta_0 + med_fint
    log_scale_trt  = beta_0 + med_fint + med_fslope
    true_s_ctrl = exp.(-(t_grid ./ exp(log_scale_ctrl)) .^ alpha)
    true_s_trt  = exp.(-(t_grid ./ exp(log_scale_trt)) .^ alpha)

    # Hospital variation band (10th-90th percentile of frailties)
    lo_fint = quantile(frailty_int, 0.1)
    hi_fint = quantile(frailty_int, 0.9)
    lo_fslope = quantile(frailty_slope, 0.1)
    hi_fslope = quantile(frailty_slope, 0.9)

    band_ctrl_lo = exp.(-(t_grid ./ exp(beta_0 + lo_fint)) .^ alpha)
    band_ctrl_hi = exp.(-(t_grid ./ exp(beta_0 + hi_fint)) .^ alpha)
    band_trt_lo  = exp.(-(t_grid ./ exp(beta_0 + hi_fint + hi_fslope)) .^ alpha)
    band_trt_hi  = exp.(-(t_grid ./ exp(beta_0 + lo_fint + lo_fslope)) .^ alpha)

    ctrl_color = parse(Colorant, "#4C9EEB")
    trt_color  = parse(Colorant, "#E05252")

    fig = Figure(size=(700, 450), backgroundcolor=th.bg)
    ax = Axis(fig[1, 1];
        xlabel = "Time",
        ylabel = "Survival Probability",
        backgroundcolor = th.bg,
        xlabelcolor = th.fg,
        ylabelcolor = th.fg,
        xticklabelcolor = th.fg,
        yticklabelcolor = th.fg,
        xtickcolor = th.muted,
        ytickcolor = th.muted,
        xgridcolor = th.grid,
        ygridcolor = th.grid,
        bottomspinecolor = th.muted,
        leftspinecolor = th.muted,
        topspinevisible = false,
        rightspinevisible = false,
        xlabelsize = 16,
        ylabelsize = 16,
        xticklabelsize = 13,
        yticklabelsize = 13,
        xlabelfont = :bold,
        ylabelfont = :bold,
    )

    # Hospital variation bands
    band!(ax, collect(t_grid), band_ctrl_lo, band_ctrl_hi;
        color = (ctrl_color, 0.15))
    band!(ax, collect(t_grid), band_trt_lo, band_trt_hi;
        color = (trt_color, 0.15))

    # KM step functions
    stairs!(ax, km_ctrl_t, km_ctrl_s;
        color = (ctrl_color, 0.5), linewidth = 1.5, step = :post,
        label = "KM: Control")
    stairs!(ax, km_trt_t, km_trt_s;
        color = (trt_color, 0.5), linewidth = 1.5, step = :post,
        label = "KM: Treatment")

    # True curves
    lines!(ax, collect(t_grid), true_s_ctrl;
        color = ctrl_color, linewidth = 2.5, label = "True: Control")
    lines!(ax, collect(t_grid), true_s_trt;
        color = trt_color, linewidth = 2.5, label = "True: Treatment")

    ylims!(ax, 0, 1.02)

    Legend(fig[1, 2], ax;
        bgcolor = :transparent,
        framevisible = false,
        labelcolor = th.fg,
        labelsize = 13,
        padding = (10, 10, 5, 5),
    )

    save(joinpath(plot_dir, "survival_curves_$(mode).svg"), fig)
    save(joinpath(plot_dir, "survival_curves_$(mode).png"), fig, px_per_unit=3)
    println("Saved survival curves ($mode)")
    return fig
end

# ── Main ─────────────────────────────────────────────────────────────────────

results = load_results()

if isempty(results)
    println("No benchmark results found in $result_dir")
    println("Run generate_data.jl, then bench_phaseskate.jl / bench_stan.R / bench_turing.jl first.")
else
    println("Found results for: $(join(keys(results), ", "))")
    println()
    for mode in [:dark, :light]
        plot_ess_bars(results; mode)
    end
end

# Survival curves only need data + true params (no benchmark results)
if isfile(joinpath(data_dir, "survival_data.json"))
    for mode in [:dark, :light]
        plot_survival_curves(; mode)
    end
else
    println("No survival data found. Run generate_data.jl first.")
end

println("\nAll plots saved to: $plot_dir")
