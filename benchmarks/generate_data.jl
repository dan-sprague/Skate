# benchmarks/generate_data.jl
# Generate the JointALM synthetic dataset for cross-language benchmarking.
# Writes benchmarks/data/joint_alm_data.json and benchmarks/data/true_params.json.
#
# Usage: julia benchmarks/generate_data.jl

using Random
using Statistics: median
using Distributions: Weibull, Exponential, BetaBinomial

# JSON writing without external deps — sufficient for numeric data
function write_json(path, obj)
    open(path, "w") do io
        _write_json(io, obj)
        println(io)
    end
end

function _write_json(io, x::AbstractDict)
    print(io, "{")
    first = true
    for (k, v) in x
        first || print(io, ",")
        first = false
        print(io, "\n  \"$k\": ")
        _write_json(io, v)
    end
    print(io, "\n}")
end

function _write_json(io, x::Real)
    if isinteger(x) && abs(x) < 1e15
        print(io, Int(x))
    else
        print(io, x)
    end
end

function _write_json(io, x::Integer)
    print(io, x)
end

function _write_json(io, x::AbstractVector)
    print(io, "[")
    for (i, v) in enumerate(x)
        i > 1 && print(io, ", ")
        _write_json(io, v)
    end
    print(io, "]")
end

function _write_json(io, x::AbstractMatrix)
    # Write as row-major array of arrays (Stan convention)
    nr, nc = size(x)
    print(io, "[")
    for i in 1:nr
        i > 1 && print(io, ",\n    ")
        print(io, "[")
        for j in 1:nc
            j > 1 && print(io, ", ")
            _write_json(io, x[i, j])
        end
        print(io, "]")
    end
    print(io, "]")
end

# ─────────────────────────────────────────────────────────────────────────────
# Ground truth parameters
# ─────────────────────────────────────────────────────────────────────────────

const TRUE = (
    log_shape      = 0.2,
    log_scale      = 2.5,
    beta_s         = [0.3, -0.2, 0.1, -0.15],
    beta_k         = [0.4, -0.3, 0.15, -0.1],
    sigma_country_k = 0.1,
    sigma_country_s = 0.5,
    mu_country_k   = [0.05, -0.08, 0.03, 0.0],
    mu_country_s   = [0.2, -0.3, 0.1, 0.0],
    mu_k           = log(0.08),
    omega_k        = 0.3,
    gamma_k        = 1.0,
    gamma_hill     = 3.0,
    EC50           = 0.4,
    log_phi        = log(15.0),
    P0             = 0.2,
)

n1, n2 = 3500, 150
p = 4
n_countries = 4
MRC_MAX = 20

# ─────────────────────────────────────────────────────────────────────────────
# Simulate data (same RNG stream as joint_alm.jl)
# ─────────────────────────────────────────────────────────────────────────────

Random.seed!(42)

shape_true = exp(TRUE.log_shape)
scale_true = exp(TRUE.log_scale)
phi_true   = exp(TRUE.log_phi)

true_z_k  = randn(n2)
_ = randn(n1)  # consume RNG state to keep data identical

tier1_X = randn(n1, p)
tier1_country_ids = rand(1:n_countries, n1)
tier2_X = randn(n2, p)
tier2_country_ids = rand(1:n_countries, n2)

# Tier 2 survival
tier2_times = Vector{Float64}(undef, n2)
true_log_k_2 = Vector{Float64}(undef, n2)
for i in 1:n2
    xbk = sum(tier2_X[i, j] * TRUE.beta_k[j] for j in 1:p)
    xbs = sum(tier2_X[i, j] * TRUE.beta_s[j] for j in 1:p)
    ce_k = TRUE.mu_country_k[tier2_country_ids[i]]
    ce_s = TRUE.mu_country_s[tier2_country_ids[i]]
    true_log_k_2[i] = TRUE.mu_k + xbk + ce_k + TRUE.omega_k * true_z_k[i]
    log_eff_scale = TRUE.log_scale - (xbs + ce_s + TRUE.gamma_k * true_log_k_2[i]) / shape_true
    tier2_times[i] = rand(Weibull(shape_true, exp(log_eff_scale)))
end
cens_times_2 = rand(Exponential(median(tier2_times) * 1.5), n2)
tier2_observed = tier2_times .<= cens_times_2
tier2_times .= min.(tier2_times, cens_times_2)
tier2_obs_idx  = findall(tier2_observed)
tier2_cens_idx = findall(.!tier2_observed)

# Tier 1 survival
tier1_times = Vector{Float64}(undef, n1)
for i in 1:n1
    xbk = sum(tier1_X[i, j] * TRUE.beta_k[j] for j in 1:p)
    xbs = sum(tier1_X[i, j] * TRUE.beta_s[j] for j in 1:p)
    ce_k = TRUE.mu_country_k[tier1_country_ids[i]]
    ce_s = TRUE.mu_country_s[tier1_country_ids[i]]
    log_k_i = TRUE.mu_k + xbk + ce_k
    log_eff_scale = TRUE.log_scale - (xbs + ce_s + TRUE.gamma_k * log_k_i) / shape_true
    tier1_times[i] = rand(Weibull(shape_true, exp(log_eff_scale)))
end
cens_times_1 = rand(Exponential(median(tier1_times) * 1.5), n1)
tier1_observed = tier1_times .<= cens_times_1
tier1_times .= min.(tier1_times, cens_times_1)
tier1_obs_idx  = findall(tier1_observed)
tier1_cens_idx = findall(.!tier1_observed)

# MRC scores (longitudinal beta-binomial)
obs_per_patient = 5
total_mrc_obs = n2 * obs_per_patient
mrc_patient_ids = repeat(1:n2, inner=obs_per_patient)
mrc_times_flat  = Float64[rand() * tier2_times[mrc_patient_ids[i]] for i in 1:total_mrc_obs]

log_P0_ratio_true = log1p(-TRUE.P0) - log(TRUE.P0)
log_EC50g_true    = TRUE.gamma_hill * log(TRUE.EC50)

mrc_scores_flat = Vector{Int}(undef, total_mrc_obs)
for i in 1:total_mrc_obs
    pid   = mrc_patient_ids[i]
    k_i   = exp(true_log_k_2[pid])
    P_t   = 1.0 / (1.0 + exp(log_P0_ratio_true - k_i * mrc_times_flat[i]))
    log_Pg = TRUE.gamma_hill * log(max(P_t, 1e-9))
    mu_mrc = clamp(1.0 / (1.0 + exp(log_Pg - log_EC50g_true)), 1e-6, 1.0 - 1e-6)
    a = mu_mrc * phi_true
    b = (1.0 - mu_mrc) * phi_true
    mrc_scores_flat[i] = rand(BetaBinomial(MRC_MAX, a, b))
end

println("Data generated from ground truth.")
println("  Tier 1: $(length(tier1_obs_idx)) observed, $(length(tier1_cens_idx)) censored")
println("  Tier 2: $(length(tier2_obs_idx)) observed, $(length(tier2_cens_idx)) censored")
println("  MRC obs: $total_mrc_obs")

# ─────────────────────────────────────────────────────────────────────────────
# Write to JSON (all indices 1-based — Julia/Stan native)
# Python loaders must subtract 1 from index arrays.
# ─────────────────────────────────────────────────────────────────────────────

data_dir = joinpath(@__DIR__, "data")
mkpath(data_dir)

data_dict = Dict{String, Any}(
    "n1" => n1,
    "n2" => n2,
    "p" => p,
    "n_countries" => n_countries,
    "MRC_MAX" => MRC_MAX,
    "tier1_times" => tier1_times,
    "tier1_X" => tier1_X,
    "tier1_country_ids" => tier1_country_ids,
    "n1_obs" => length(tier1_obs_idx),
    "n1_cens" => length(tier1_cens_idx),
    "tier1_obs_idx" => tier1_obs_idx,
    "tier1_cens_idx" => tier1_cens_idx,
    "tier2_times" => tier2_times,
    "tier2_X" => tier2_X,
    "tier2_country_ids" => tier2_country_ids,
    "n2_obs" => length(tier2_obs_idx),
    "n2_cens" => length(tier2_cens_idx),
    "tier2_obs_idx" => tier2_obs_idx,
    "tier2_cens_idx" => tier2_cens_idx,
    "total_mrc_obs" => total_mrc_obs,
    "mrc_scores_flat" => mrc_scores_flat,
    "mrc_times_flat" => mrc_times_flat,
    "mrc_patient_ids" => mrc_patient_ids,
)

write_json(joinpath(data_dir, "joint_alm_data.json"), data_dict)
println("\nWrote: benchmarks/data/joint_alm_data.json")

# Ground truth params
true_dict = Dict{String, Any}(
    "log_shape" => TRUE.log_shape,
    "log_scale" => TRUE.log_scale,
    "beta_s" => TRUE.beta_s,
    "beta_k" => TRUE.beta_k,
    "sigma_country_k" => TRUE.sigma_country_k,
    "sigma_country_s" => TRUE.sigma_country_s,
    "mu_country_k" => TRUE.mu_country_k,
    "mu_country_s" => TRUE.mu_country_s,
    "mu_k" => TRUE.mu_k,
    "omega_k" => TRUE.omega_k,
    "gamma_k" => TRUE.gamma_k,
    "gamma_hill" => TRUE.gamma_hill,
    "EC50" => TRUE.EC50,
    "log_phi" => TRUE.log_phi,
    "P0" => TRUE.P0,
    "z_k" => true_z_k,
)

write_json(joinpath(data_dir, "true_params.json"), true_dict)
println("Wrote: benchmarks/data/true_params.json")
