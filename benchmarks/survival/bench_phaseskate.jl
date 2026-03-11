# benchmarks/survival/bench_phaseskate.jl
# Benchmark PhaseSkate on the Weibull survival frailty model.
#
# Usage: julia benchmarks/survival/bench_phaseskate.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using PhaseSkate
using Printf
using Random
using Statistics: median

# ── Configuration ────────────────────────────────────────────────────────────

const NUM_CHAINS  = 4
const NUM_WARMUP  = 1000
const NUM_SAMPLES = 2000
const SEED        = 42

# ── Generate data inline ────────────────────────────────────────────────────

Random.seed!(42)

const H = 100; const N = 5000; const P = 8

# True parameters
true_log_alpha = 0.4
true_beta_0 = 2.5
true_beta = [0.3, -0.4, 0.2, -0.15, 0.25, -0.1, 0.35, -0.2]
true_log_sigma_int = -0.5
true_log_sigma_slope = -0.8
true_rho = 0.4

alpha = exp(true_log_alpha)
sigma_int = exp(true_log_sigma_int)
sigma_slope = exp(true_log_sigma_slope)
sqrt_1mrho2 = sqrt(1.0 - true_rho^2)

z_int = randn(H)
z_slope = randn(H)
frailty_int = sigma_int .* z_int
frailty_slope = sigma_slope .* (true_rho .* z_int .+ sqrt_1mrho2 .* z_slope)

X = randn(N, P)
trt = Float64.(rand(0:1, N))
hosp = rand(1:H, N)

log_scale = Vector{Float64}(undef, N)
for i in 1:N
    h = hosp[i]
    xb = sum(X[i, j] * true_beta[j] for j in 1:P)
    log_scale[i] = true_beta_0 + xb + frailty_int[h] + trt[i] * frailty_slope[h]
end

using Distributions: Weibull, Exponential
times = [rand(Weibull(alpha, exp(log_scale[i]))) for i in 1:N]
cens_times = rand(Exponential(median(times) * 1.5), N)
observed = times .<= cens_times
times .= min.(times, cens_times)
obs_idx = findall(observed)
cens_idx = findall(.!observed)

println("Data: N=$N, P=$P, H=$H, observed=$(length(obs_idx)), censored=$(length(cens_idx))")

# ── Model definition ────────────────────────────────────────────────────────

@skate SurvivalFrailty begin
    @constants begin
        N::Int; P::Int; H::Int
        X::Matrix{Float64}; trt::Vector{Float64}; times::Vector{Float64}
        hosp::Vector{Int}; obs_idx::Vector{Int}; cens_idx::Vector{Int}
    end
    @params begin
        log_alpha::Float64; beta_0::Float64
        beta = param(Vector{Float64}, P)
        log_sigma_int::Float64; log_sigma_slope::Float64
        rho = param(Float64; lower=-1.0, upper=1.0)
        z_int = param(Vector{Float64}, H); z_slope = param(Vector{Float64}, H)
    end
    @logjoint begin
        alpha = exp(log_alpha)
        sigma_int = exp(log_sigma_int)
        sigma_slope = exp(log_sigma_slope)
        sqrt_1mrho2 = sqrt(1.0 - rho * rho)
        target += normal_lpdf(log_alpha, 0.0, 0.5)
        target += normal_lpdf(beta_0, 2.0, 2.0)
        target += multi_normal_diag_lpdf(beta, 0.0, 1.0)
        target += normal_lpdf(log_sigma_int, -1.0, 1.0)
        target += normal_lpdf(log_sigma_slope, -1.0, 1.0)
        target += log(1.0 - rho * rho)
        target += multi_normal_diag_lpdf(z_int, 0.0, 1.0)
        target += multi_normal_diag_lpdf(z_slope, 0.0, 1.0)
        @for log_scale = beta_0 .+ (X * beta) .+ sigma_int .* z_int[hosp] .+ trt .* sigma_slope .* (rho .* z_int[hosp] .+ sqrt_1mrho2 .* z_slope[hosp])
        target += weibull_logsigma_lpdf_sum(times, alpha, log_scale, obs_idx)
        target += weibull_logsigma_lccdf_sum(times, alpha, log_scale, cens_idx)
    end
end

# ── Build model ──────────────────────────────────────────────────────────────

d = SurvivalFrailtyData(
    N=N, P=P, H=H, X=X, trt=trt, times=times,
    hosp=hosp, obs_idx=obs_idx, cens_idx=cens_idx)

m = make(d)
println("SurvivalFrailty Model - dim=$(m.dim)")

# ── Compile gradient (excluded from timing) ──────────────────────────────────

println("\nCompiling gradient...")
_ = sample(m, 1; warmup=1, chains=1, seed=0)
println("Compilation done.\n")

# ── Sample ───────────────────────────────────────────────────────────────────

println("Sampling: $NUM_WARMUP warmup, $NUM_SAMPLES samples, $NUM_CHAINS chains")

t_start = time()
ch = sample(m, NUM_SAMPLES; warmup=NUM_WARMUP, chains=NUM_CHAINS, seed=SEED, metric=:dense)
t_total = time() - t_start

# ── Diagnostics ──────────────────────────────────────────────────────────────

display(ch)

ess_min = min_ess(ch)

println("\n", "-- Results ", "-"^39)
@printf("  Total wall time:  %.1f s\n", t_total)
@printf("  Min ESS (bulk):   %.0f\n", ess_min)
@printf("  ESS/s:            %.1f\n", ess_min / t_total)
