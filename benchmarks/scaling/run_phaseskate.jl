using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using PhaseSkate, Random, Statistics, Printf
using Distributions: Weibull, Exponential
using JSON3
import PhaseSkate: sample

const P = 8
const N_PER_H = 50  # observations per hospital
const TRUE_BETA = [0.3, -0.4, 0.2, -0.15, 0.25, -0.1, 0.35, -0.2]
const TRUE_TRT_EFFECT = -0.75  # fixed treatment effect on log-scale (HR ≈ exp(-alpha * trt_effect))

function generate_data(H; seed=42)
    Random.seed!(seed)
    N = H * N_PER_H
    alpha = exp(0.4); sigma_int = exp(-0.5); sigma_slope = exp(-0.8)
    sqrt_1mrho2 = sqrt(1.0 - 0.4^2)
    z_int = randn(H); z_slope = randn(H)
    frailty_int = sigma_int .* z_int
    frailty_slope = sigma_slope .* (0.4 .* z_int .+ sqrt_1mrho2 .* z_slope)
    X = randn(N, P); trt = Float64.(rand(0:1, N)); hosp = repeat(1:H, inner=N_PER_H)
    log_scale = [2.5 + sum(X[i,j]*TRUE_BETA[j] for j in 1:P) + frailty_int[hosp[i]] + trt[i]*(TRUE_TRT_EFFECT + frailty_slope[hosp[i]]) for i in 1:N]
    times = [rand(Weibull(alpha, exp(log_scale[i]))) for i in 1:N]
    cens_times = rand(Exponential(median(times)*1.5), N)
    observed = times .<= cens_times
    times .= min.(times, cens_times)
    obs_idx = findall(observed)
    cens_idx = findall(.!observed)

    # Save data as JSON for Stan
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    json_data = Dict(
        "N" => N, "P" => P, "H" => H,
        "X" => [X[i,j] for i in 1:N, j in 1:P] |> x -> reshape(x, N, P) |> eachrow .|> collect,
        "trt" => trt, "times" => times,
        "hosp" => hosp,
        "N_obs" => length(obs_idx), "N_cens" => length(cens_idx),
        "obs_idx" => obs_idx, "cens_idx" => cens_idx
    )
    open(joinpath(data_dir, "data_H$(H).json"), "w") do f
        JSON3.write(f, json_data)
    end

    return (N=N, P=P, H=H, X=X, trt=trt, times=times, hosp=hosp,
            obs_idx=obs_idx, cens_idx=cens_idx, observed=observed)
end

@skate SurvivalFrailty begin
    @constants begin; N::Int; P::Int; H::Int
        X::Matrix{Float64}; trt::Vector{Float64}; times::Vector{Float64}
        hosp::Vector{Int}; obs_idx::Vector{Int}; cens_idx::Vector{Int}; end
    @params begin; log_alpha::Float64; beta_0::Float64
        trt_effect::Float64
        beta = param(Vector{Float64}, P)
        log_sigma_int::Float64; log_sigma_slope::Float64
        rho = param(Float64; lower=-1.0, upper=1.0)
        z_int = param(Vector{Float64}, H); z_slope = param(Vector{Float64}, H); end
    @logjoint begin
        alpha = exp(log_alpha); sigma_int = exp(log_sigma_int)
        sigma_slope = exp(log_sigma_slope); sqrt_1mrho2 = sqrt(1.0 - rho*rho)
        target += normal_lpdf(log_alpha, 0.0, 0.5); target += normal_lpdf(beta_0, 2.0, 2.0)
        target += normal_lpdf(trt_effect, 0.0, 1.0)
        target += multi_normal_diag_lpdf(beta, 0.0, 1.0)
        target += normal_lpdf(log_sigma_int, -1.0, 1.0); target += normal_lpdf(log_sigma_slope, -1.0, 1.0)
        target += log(1.0 - rho*rho)
        target += multi_normal_diag_lpdf(z_int, 0.0, 1.0); target += multi_normal_diag_lpdf(z_slope, 0.0, 1.0)
        @for log_scale = beta_0 .+ (X*beta) .+ sigma_int .* z_int[hosp] .+ trt .* (trt_effect .+ sigma_slope .* (rho .* z_int[hosp] .+ sqrt_1mrho2 .* z_slope[hosp]))
        target += weibull_logsigma_lpdf_sum(times, alpha, log_scale, obs_idx)
        target += weibull_logsigma_lccdf_sum(times, alpha, log_scale, cens_idx)
    end
end

function run_benchmark(H; metric=:dense, num_samples=2000, warmup=1000, chains=4, seed=42)
    data = generate_data(H; seed)
    d = SurvivalFrailtyData(N=data.N, P=data.P, H=data.H, X=data.X, trt=data.trt,
                             times=data.times, hosp=data.hosp,
                             obs_idx=data.obs_idx, cens_idx=data.cens_idx)
    m = make(d)
    dim = m.dim

    # Warmup JIT
    _ = sample(m, 1; warmup=1, chains=1, seed=0)

    t0 = time()
    ch = sample(m, num_samples; warmup, chains, seed, metric)
    wall = time() - t0

    diag = diagnostics(ch)
    ess_vals = diag.ess_bulk
    min_ess = minimum(ess_vals)
    med_ess = median(ess_vals)
    ess_per_s = min_ess / wall

    @printf("\n=== %s  H=%d  dim=%d ===\n", metric, H, dim)
    @printf("  Min ESS:    %.1f\n", min_ess)
    @printf("  Median ESS: %.1f\n", med_ess)
    @printf("  Wall time:  %.1f s\n", wall)
    @printf("  ESS/s:      %.1f\n", ess_per_s)

    return Dict("H" => H, "dim" => dim, "min_ess" => min_ess,
                "median_ess" => med_ess, "wall_time" => wall, "ess_per_s" => ess_per_s)
end

# Parse metric from CLI: julia run_phaseskate.jl [dense|diagonal]
metric_arg = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :dense
suffix = metric_arg == :dense ? "" : "_diagonal"

H_values = [10, 25, 50, 100]
results = Dict[]
for H in H_values
    push!(results, run_benchmark(H; metric=metric_arg))
end

# Save results
outfile = joinpath(@__DIR__, "results_phaseskate$(suffix).json")
open(outfile, "w") do f
    JSON3.write(f, results)
end
println("\nResults saved to $outfile")
