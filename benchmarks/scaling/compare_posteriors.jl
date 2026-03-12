# Compare PhaseSkate vs Stan posterior estimates on the H=100 survival model.
# Reports max absolute difference in means, 95% CI, and 99% CI across all parameters.

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using PhaseSkate, Random, Statistics, Printf
using Distributions: Weibull, Exponential
using JSON3
import PhaseSkate: sample

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
            obs_idx=obs_idx, cens_idx=cens_idx, observed=observed)
end

@skate SurvivalFrailty begin
    @constants begin; N::Int; P::Int; H::Int
        X::Matrix{Float64}; trt::Vector{Float64}; times::Vector{Float64}
        hosp::Vector{Int}; obs_idx::Vector{Int}; cens_idx::Vector{Int}; end
    @params begin; log_alpha::Float64; beta_0::Float64
        beta = param(Vector{Float64}, P)
        log_sigma_int::Float64; log_sigma_slope::Float64
        rho = param(Float64; lower=-1.0, upper=1.0)
        z_int = param(Vector{Float64}, H); z_slope = param(Vector{Float64}, H); end
    @logjoint begin
        alpha = exp(log_alpha); sigma_int = exp(log_sigma_int)
        sigma_slope = exp(log_sigma_slope); sqrt_1mrho2 = sqrt(1.0 - rho*rho)
        target += normal_lpdf(log_alpha, 0.0, 0.5); target += normal_lpdf(beta_0, 2.0, 2.0)
        target += multi_normal_diag_lpdf(beta, 0.0, 1.0)
        target += normal_lpdf(log_sigma_int, -1.0, 1.0); target += normal_lpdf(log_sigma_slope, -1.0, 1.0)
        target += log(1.0 - rho*rho)
        target += multi_normal_diag_lpdf(z_int, 0.0, 1.0); target += multi_normal_diag_lpdf(z_slope, 0.0, 1.0)
        @for log_scale = beta_0 .+ (X*beta) .+ sigma_int .* z_int[hosp] .+ trt .* sigma_slope .* (rho .* z_int[hosp] .+ sqrt_1mrho2 .* z_slope[hosp])
        target += weibull_logsigma_lpdf_sum(times, alpha, log_scale, obs_idx)
        target += weibull_logsigma_lccdf_sum(times, alpha, log_scale, cens_idx)
    end
end

# ── Run PhaseSkate ──

const H_VAL = 100
data = generate_data(H_VAL; seed=42)
d = SurvivalFrailtyData(N=data.N, P=data.P, H=data.H, X=data.X, trt=data.trt,
                         times=data.times, hosp=data.hosp,
                         obs_idx=data.obs_idx, cens_idx=data.cens_idx)
m = make(d)

# Warmup JIT
_ = sample(m, 1; warmup=1, chains=1, seed=0)

# Target min ESS ≈ 2000 for fair comparison.
# PS dense at H=100 gets min ESS=2584 from 2000 draws/chain → scale to ~1548.
const TARGET_ESS = 2000
const PS_DRAWS = 1548
ch = sample(m, PS_DRAWS; warmup=1000, chains=4, seed=42, metric=:dense)

# ── Collect PhaseSkate summaries ──

ps_params = Dict{String, NamedTuple}()
for p in ch.params
    name = string(p.name)
    μ = mean(ch, p.name)
    lo95, hi95 = ci(ch, p.name; level=0.95)
    lo99, hi99 = ci(ch, p.name; level=0.99)
    if p.shape == ()
        ps_params[name] = (mean=μ, lo95=lo95, hi95=hi95, lo99=lo99, hi99=hi99)
    else
        for i in eachindex(μ isa AbstractArray ? μ : Float64[μ])
            k = "$(name)[$(i)]"
            ps_params[k] = (mean=μ[i], lo95=lo95[i], hi95=hi95[i], lo99=lo99[i], hi99=hi99[i])
        end
    end
end

# ── Load Stan draws from CSVs ──

function load_stan_draws(H)
    dir = joinpath(@__DIR__, "stan_output")
    chains = Matrix{Float64}[]
    header = String[]
    for c in 1:4
        f = joinpath(dir, "output_H$(H)_chain$(c).csv")
        lines = readlines(f)
        # Skip comment lines
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
    return header, chains
end

header, stan_chains = load_stan_draws(H_VAL)

# Truncate Stan draws to target ESS ≈ 2000.
# Stan dense at H=100 gets min ESS=2993 from 2000 draws/chain → keep first ~1337.
const STAN_DRAWS = 1337
stan_chains = [c[1:min(STAN_DRAWS, size(c,1)), :] for c in stan_chains]

# Pool all Stan chains
stan_pooled = vcat(stan_chains...)

stan_params = Dict{String, NamedTuple}()
stan_sd = Dict{String, Float64}()
for (j, name) in enumerate(header)
    endswith(name, "__") && continue
    draws = stan_pooled[:, j]
    μ = Statistics.mean(draws)
    σ = Statistics.std(draws)
    lo95 = quantile(draws, 0.025)
    hi95 = quantile(draws, 0.975)
    lo99 = quantile(draws, 0.005)
    hi99 = quantile(draws, 0.995)

    # Stan: beta.1 → beta[1], z_int.1 → z_int[1], etc.
    ps_name = replace(name, r"\.(\d+)" => s"[\1]")
    stan_params[ps_name] = (mean=μ, lo95=lo95, hi95=hi95, lo99=lo99, hi99=hi99)
    stan_sd[ps_name] = σ
end

# ── Compare (normalized by posterior SD) ──

common = sort(collect(intersect(keys(ps_params), keys(stan_params))))

println("\n" * "="^80)
println("Posterior Comparison: PhaseSkate vs Stan (H=$H_VAL, dim=$(m.dim))")
println("All differences normalized by Stan's posterior SD")
println("="^80)

# Collect signed normalized differences: (PhaseSkate - Stan) / Stan_SD
diff_mean = Float64[]
diff_lo95 = Float64[]
diff_hi95 = Float64[]
diff_lo99 = Float64[]
diff_hi99 = Float64[]

for k in common
    ps = ps_params[k]
    st = stan_params[k]
    σ = max(stan_sd[k], 1e-10)

    push!(diff_mean, (ps.mean - st.mean) / σ)
    push!(diff_lo95, (ps.lo95 - st.lo95) / σ)
    push!(diff_hi95, (ps.hi95 - st.hi95) / σ)
    push!(diff_lo99, (ps.lo99 - st.lo99) / σ)
    push!(diff_hi99, (ps.hi99 - st.hi99) / σ)
end

# Top 10 worst by |normalized mean difference|
idx = sortperm(abs.(diff_mean); rev=true)
println("\nTop 10 largest differences in posterior mean (in posterior SDs):")
println("-"^90)
@printf("%-20s %10s %10s %8s %11s %10s\n",
        "Parameter", "PS mean", "Stan mean", "Stan SD", "(PS-Stan)/σ", "95% CI width")
println("-"^90)
for i in idx[1:min(10, length(idx))]
    k = common[i]
    ps = ps_params[k]
    st = stan_params[k]
    σ = stan_sd[k]
    ci_width = st.hi95 - st.lo95
    @printf("%-20s %10.4f %10.4f %8.4f %+10.3f σ %10.4f\n",
            k, ps.mean, st.mean, σ, diff_mean[i], ci_width)
end

# Summary table — signed differences show bias (or lack thereof)
println("\n" * "="^80)
@printf("Summary across %d parameters: (PhaseSkate - Stan) / Stan_SD\n", length(common))
println("="^80)

function fmt_signed(label, vals)
    @printf("  %-30s  mean=%+7.4f σ   median=%+7.4f σ   |max|=%6.3f σ   std=%6.4f σ\n",
            label, Statistics.mean(vals), median(vals), maximum(abs.(vals)), Statistics.std(vals))
end

fmt_signed("Δ mean / σ", diff_mean)
fmt_signed("Δ 95% CI lower / σ", diff_lo95)
fmt_signed("Δ 95% CI upper / σ", diff_hi95)
fmt_signed("Δ 99% CI lower / σ", diff_lo99)
fmt_signed("Δ 99% CI upper / σ", diff_hi99)

# Save as JSON
results = Dict(
    "n_params" => length(common),
    "note" => "Signed differences (PS - Stan) normalized by Stan posterior SD",
    "diff_mean" => Dict("mean" => Statistics.mean(diff_mean), "median" => median(diff_mean), "std" => Statistics.std(diff_mean), "abs_max" => maximum(abs.(diff_mean))),
    "diff_95ci_lower" => Dict("mean" => Statistics.mean(diff_lo95), "median" => median(diff_lo95), "std" => Statistics.std(diff_lo95), "abs_max" => maximum(abs.(diff_lo95))),
    "diff_95ci_upper" => Dict("mean" => Statistics.mean(diff_hi95), "median" => median(diff_hi95), "std" => Statistics.std(diff_hi95), "abs_max" => maximum(abs.(diff_hi95))),
    "diff_99ci_lower" => Dict("mean" => Statistics.mean(diff_lo99), "median" => median(diff_lo99), "std" => Statistics.std(diff_lo99), "abs_max" => maximum(abs.(diff_lo99))),
    "diff_99ci_upper" => Dict("mean" => Statistics.mean(diff_hi99), "median" => median(diff_hi99), "std" => Statistics.std(diff_hi99), "abs_max" => maximum(abs.(diff_hi99))),
)
open(joinpath(@__DIR__, "posterior_comparison.json"), "w") do f
    JSON3.write(f, results)
end
println("\nResults saved to benchmarks/scaling/posterior_comparison.json")

# Save per-parameter means for scatter plot
per_param = Dict{String, Any}[]
for k in common
    ps = ps_params[k]
    st = stan_params[k]
    push!(per_param, Dict("name" => k, "ps_mean" => ps.mean, "stan_mean" => st.mean, "stan_sd" => stan_sd[k]))
end
open(joinpath(@__DIR__, "posterior_means.json"), "w") do f
    JSON3.write(f, per_param)
end
println("Per-parameter means saved to benchmarks/scaling/posterior_means.json")
