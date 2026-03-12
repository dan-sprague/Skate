# Extract per-parameter posterior means from PS and Stan for the scatter plot.
# Runs PS sampling once (no benchmarking), reads Stan from CSVs.
# Output: posterior_means.json

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using PhaseSkate, Random, Statistics, JSON3
using Distributions: Weibull, Exponential
import PhaseSkate: sample

const P = 8
const N_PER_H = 50
const H = 100
const TRUE_BETA = [0.3, -0.4, 0.2, -0.15, 0.25, -0.1, 0.35, -0.2]
const TRUE_TRT_EFFECT = -0.75

function generate_data(; seed=42)
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
    return (N=N, P=P, H=H, X=X, trt=trt, times=times, hosp=hosp,
            obs_idx=obs_idx, cens_idx=cens_idx)
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

data = generate_data()
d = SurvivalFrailtyData(N=data.N, P=data.P, H=data.H, X=data.X, trt=data.trt,
                         times=data.times, hosp=data.hosp,
                         obs_idx=data.obs_idx, cens_idx=data.cens_idx)
m = make(d)

println("Sampling PS (1548 draws × 4 chains, dense metric)...")
ch = sample(m, 1548; warmup=1000, chains=4, seed=42, metric=:dense)

# Collect PS means
ps_means = Dict{String, Float64}()
for p in ch.params
    name = string(p.name)
    μ = mean(ch, p.name)
    if p.shape == ()
        ps_means[name] = μ
    else
        for i in eachindex(μ isa AbstractArray ? μ : Float64[μ])
            ps_means["$(name)[$(i)]"] = μ[i]
        end
    end
end

# Load Stan means
dir = joinpath(@__DIR__, "stan_output")
stan_chains_data = Matrix{Float64}[]
local header = String[]
for c in 1:4
    f = joinpath(dir, "output_H$(H)_chain$(c).csv")
    lines = readlines(f)
    data_lines = filter(l -> !startswith(l, '#'), lines)
    global header = split(data_lines[1], ',')
    nrows = length(data_lines) - 1
    ncols = length(header)
    mat = Matrix{Float64}(undef, nrows, ncols)
    for (i, line) in enumerate(data_lines[2:end])
        mat[i, :] .= parse.(Float64, split(line, ','))
    end
    push!(stan_chains_data, mat)
end
# Truncate Stan to 1337 draws per chain for ESS ≈ 2000
stan_chains_data = [c[1:min(1337, size(c,1)), :] for c in stan_chains_data]
pooled = vcat(stan_chains_data...)

stan_means = Dict{String, Float64}()
stan_sds = Dict{String, Float64}()
for (j, name) in enumerate(header)
    endswith(name, "__") && continue
    ps_name = replace(string(name), r"\.(\d+)" => s"[\1]")
    stan_means[ps_name] = Statistics.mean(pooled[:, j])
    stan_sds[ps_name] = Statistics.std(pooled[:, j])
end

# Match and save
common = sort(collect(intersect(keys(ps_means), keys(stan_means))))
per_param = [Dict("name" => k, "ps_mean" => ps_means[k], "stan_mean" => stan_means[k], "stan_sd" => stan_sds[k]) for k in common]

outfile = joinpath(@__DIR__, "posterior_means.json")
open(outfile, "w") do f; JSON3.write(f, per_param); end
println("Saved $outfile ($(length(common)) parameters)")
