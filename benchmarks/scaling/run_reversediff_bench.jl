# Benchmark: ReverseDiff (compiled tape) vs Enzyme on the same survival model & sampler.
# Hand-writes the log-density with generic typing so ReverseDiff TrackedReals pass through,
# then calls PhaseSkate's sample() with ad=:reversediff vs ad=:reverse.
#
# Usage: julia -t 5 benchmarks/scaling/run_reversediff_bench.jl [dense|diagonal]

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using PhaseSkate, Random, Statistics, Printf, LinearAlgebra
using Distributions: Weibull, Exponential
using JSON3
import PhaseSkate: sample, ModelLogDensity, log_prob

const P = 8
const N_PER_H = 50
const TRUE_BETA = [0.3, -0.4, 0.2, -0.15, 0.25, -0.1, 0.35, -0.2]

function generate_data(H; seed=42)
    Random.seed!(seed)
    N = H * N_PER_H
    alpha = exp(0.4); sigma_int = exp(-0.5); sigma_slope = exp(-0.8)
    sqrt_1mrho2 = sqrt(1.0 - 0.4^2)
    z_int = randn(H); z_slope = randn(H)
    frailty_int = sigma_int .* z_int
    frailty_slope = sigma_slope .* (0.4 .* z_int .+ sqrt_1mrho2 .* z_slope)
    X = randn(N, P); trt = Float64.(rand(0:1, N)); hosp = repeat(1:H, inner=N_PER_H)
    log_scale = [2.5 + sum(X[i,j]*TRUE_BETA[j] for j in 1:P) + frailty_int[hosp[i]] + trt[i]*frailty_slope[hosp[i]] for i in 1:N]
    times = [rand(Weibull(alpha, exp(log_scale[i]))) for i in 1:N]
    cens_times = rand(Exponential(median(times)*1.5), N)
    observed = times .<= cens_times
    times .= min.(times, cens_times)
    obs_idx = findall(observed)
    cens_idx = findall(.!observed)
    return (N=N, P=P, H=H, X=X, trt=trt, times=times, hosp=hosp,
            obs_idx=obs_idx, cens_idx=cens_idx)
end

# ── Hand-written log-density (generic types for ReverseDiff TrackedReals) ──
# Matches @skate SurvivalFrailty exactly: same math, same parameter layout.
# q layout: [log_alpha, beta_0, beta[1:P], log_sigma_int, log_sigma_slope, rho_raw, z_int[1:H], z_slope[1:H]]

function make_survival_model(data)
    N, _P, H = data.N, data.P, data.H
    X, trt, times = data.X, data.trt, data.times
    hosp, obs_idx, cens_idx = data.hosp, data.obs_idx, data.cens_idx
    dim = 13 + 2 * H

    _logistic(x) = inv(one(x) + exp(-x))

    function logdensity(q)
        T = eltype(q)

        log_alpha       = q[1]
        beta_0          = q[2]
        beta            = @view q[3:2+_P]
        log_sigma_int   = q[3+_P]
        log_sigma_slope = q[4+_P]
        rho_raw         = q[5+_P]
        z_int           = @view q[6+_P:5+_P+H]
        z_slope         = @view q[6+_P+H:5+_P+2H]

        # Bounded transform for rho ∈ (-1, 1)
        s_rho = _logistic(rho_raw)
        rho = T(-1) + T(2) * s_rho
        log_jac = log(T(2)) + log(s_rho) + log(one(T) - s_rho)

        alpha = exp(log_alpha)
        sigma_int = exp(log_sigma_int)
        sigma_slope = exp(log_sigma_slope)
        sqrt_1mrho2 = sqrt(one(T) - rho * rho)

        target = zero(T)

        # Priors — normal_lpdf(x, μ, σ) = -0.5*((x-μ)/σ)^2 - log(σ) - 0.5*log(2π)
        target += -log_alpha^2 / (2 * T(0.5)^2) - log(T(0.5)) - T(0.5) * log(T(2π))
        target += -(beta_0 - T(2))^2 / (2 * T(2)^2) - log(T(2)) - T(0.5) * log(T(2π))

        # multi_normal_diag(beta, 0, 1)
        ss_beta = zero(T)
        for j in 1:_P
            ss_beta += beta[j]^2
        end
        target += -T(0.5) * ss_beta - T(0.5) * _P * log(T(2π))

        target += -(log_sigma_int + one(T))^2 / T(2) - T(0.5) * log(T(2π))
        target += -(log_sigma_slope + one(T))^2 / T(2) - T(0.5) * log(T(2π))

        # log(1 - rho^2) prior on correlation
        target += log(one(T) - rho * rho)

        # multi_normal_diag(z_int, 0, 1) and (z_slope, 0, 1)
        ss_zint = zero(T)
        ss_zslope = zero(T)
        for j in 1:H
            ss_zint += z_int[j]^2
            ss_zslope += z_slope[j]^2
        end
        target += -T(0.5) * ss_zint - T(0.5) * H * log(T(2π))
        target += -T(0.5) * ss_zslope - T(0.5) * H * log(T(2π))

        # Weibull likelihood — observed events
        for j in eachindex(obs_idx)
            idx = obs_idx[j]
            ls = beta_0
            for k in 1:_P
                ls += X[idx, k] * beta[k]
            end
            ls += sigma_int * z_int[hosp[idx]]
            ls += trt[idx] * sigma_slope * (rho * z_int[hosp[idx]] + sqrt_1mrho2 * z_slope[hosp[idx]])

            x_safe = max(times[idx], eps(T))
            log_x = log(x_safe)
            target += log(alpha) - ls + (alpha - one(T)) * (log_x - ls) - exp(alpha * (log_x - ls))
        end

        # Weibull likelihood — censored events
        for j in eachindex(cens_idx)
            idx = cens_idx[j]
            ls = beta_0
            for k in 1:_P
                ls += X[idx, k] * beta[k]
            end
            ls += sigma_int * z_int[hosp[idx]]
            ls += trt[idx] * sigma_slope * (rho * z_int[hosp[idx]] + sqrt_1mrho2 * z_slope[hosp[idx]])

            x_safe = max(times[idx], zero(T))
            target += -exp(alpha * (log(x_safe) - ls))
        end

        return target + log_jac
    end

    constrain = function(q::AbstractVector{Float64})
        s_rho = inv(1.0 + exp(-q[5+_P]))
        rho = -1.0 + 2.0 * s_rho
        (log_alpha = q[1], beta_0 = q[2], beta = q[3:2+_P],
         log_sigma_int = q[3+_P], log_sigma_slope = q[4+_P],
         rho = rho, z_int = q[6+_P:5+_P+H], z_slope = q[6+_P+H:5+_P+2H])
    end

    return ModelLogDensity(dim, logdensity, constrain)
end

# ── Benchmark runner ──

function run_benchmark(H; metric=:dense, ad=:reverse, num_samples=2000, warmup=1000, chains=4, seed=42)
    data = generate_data(H; seed)
    m = make_survival_model(data)
    dim = m.dim

    ad_label = ad == :reversediff ? "ReverseDiff" : "Enzyme"
    @printf("Building %s model H=%d dim=%d (ad=%s)...\n", metric, H, dim, ad_label)

    # JIT warmup
    _ = sample(m, 1; warmup=1, chains=1, seed=0, ad)

    t0 = time()
    ch = sample(m, num_samples; warmup, chains, seed, metric, ad)
    wall = time() - t0

    diag = diagnostics(ch)
    ess_vals = diag.ess_bulk
    min_ess = minimum(ess_vals)
    med_ess = median(ess_vals)
    ess_per_s = min_ess / wall

    @printf("\n=== %s %s  H=%d  dim=%d ===\n", ad_label, metric, H, dim)
    @printf("  Min ESS:    %.1f\n", min_ess)
    @printf("  Median ESS: %.1f\n", med_ess)
    @printf("  Wall time:  %.1f s\n", wall)
    @printf("  ESS/s:      %.1f\n", ess_per_s)

    return Dict("H" => H, "dim" => dim, "min_ess" => min_ess,
                "median_ess" => med_ess, "wall_time" => wall, "ess_per_s" => ess_per_s)
end

# ── Main ──

metric_arg = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :dense
suffix = metric_arg == :dense ? "" : "_diagonal"

H_values = [10, 25, 50, 100]

# Run Enzyme first (baseline)
println("="^60)
println("Running Enzyme (baseline)")
println("="^60)
results_enzyme = Dict[]
for H in H_values
    push!(results_enzyme, run_benchmark(H; metric=metric_arg, ad=:reverse))
end

outfile = joinpath(@__DIR__, "results_enzyme_handwritten$(suffix).json")
open(outfile, "w") do f; JSON3.write(f, results_enzyme); end
println("\nSaved $outfile")

# Run ReverseDiff
println("\n" * "="^60)
println("Running ReverseDiff (compiled tape)")
println("="^60)
results_rd = Dict[]
for H in H_values
    push!(results_rd, run_benchmark(H; metric=metric_arg, ad=:reversediff))
end

outfile = joinpath(@__DIR__, "results_reversediff$(suffix).json")
open(outfile, "w") do f; JSON3.write(f, results_rd); end
println("\nSaved $outfile")

# Summary comparison
println("\n" * "="^60)
println("Enzyme vs ReverseDiff Comparison ($metric_arg metric)")
println("="^60)
@printf("%-5s %5s | %10s %10s | %10s %10s | %8s\n",
    "H", "dim", "Enz wall", "Enz ESS/s",
    "RD wall", "RD ESS/s", "Slowdown")
println("-"^75)
for i in eachindex(H_values)
    e = results_enzyme[i]
    r = results_rd[i]
    slowdown = r["wall_time"] / e["wall_time"]
    @printf("%-5d %5d | %9.1fs %9.1f | %9.1fs %9.1f | %7.1fx\n",
        e["H"], e["dim"],
        e["wall_time"], e["ess_per_s"],
        r["wall_time"], r["ess_per_s"],
        slowdown)
end
