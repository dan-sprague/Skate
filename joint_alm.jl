# Joint ALM Model — @for syntax
# Uses @for broadcast-to-loop unrolling for readable, zero-allocation regression blocks.
using Pkg; Pkg.activate(@__DIR__)
using PhaseSkate
using Random
using Statistics
using Distributions: Weibull, Exponential, BetaBinomial
# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

@skate JointALM begin
    @constants begin
        n1::Int
        n2::Int
        p::Int
        n_countries::Int
        MRC_MAX::Int
        tier1_times::Vector{Float64}
        tier1_X::Matrix{Float64}
        tier1_country_ids::Vector{Int}
        tier1_obs_idx::Vector{Int}
        tier1_cens_idx::Vector{Int}
        tier2_times::Vector{Float64}
        tier2_X::Matrix{Float64}
        tier2_country_ids::Vector{Int}
        tier2_obs_idx::Vector{Int}
        tier2_cens_idx::Vector{Int}
        total_mrc_obs::Int
        mrc_scores_flat::Vector{Int}
        mrc_times_flat::Vector{Float64}
        mrc_patient_ids::Vector{Int}
    end

    @params begin
        log_shape::Float64
        log_scale::Float64
        beta_s     = param(Vector{Float64}, p)
        beta_k     = param(Vector{Float64}, p)
        sigma_country_k = param(Float64; lower=0.0)
        sigma_country_s = param(Float64; lower=0.0)
        mu_country_k = param(Vector{Float64}, n_countries)
        mu_country_s = param(Vector{Float64}, n_countries)
        mu_k::Float64
        omega_k    = param(Float64; lower=0.0)
        gamma_k::Float64
        gamma_hill = param(Float64; lower=1.0)
        EC50       = param(Float64; lower=0.0, upper=1.0)
        log_phi::Float64
        P0         = param(Float64; lower=0.0, upper=1.0)
        z_k        = param(Vector{Float64}, n2)
    end

    @logjoint begin
        # ── Derived scalars  ──
        shape = exp(log_shape)
        inv_shape = 1.0 / shape
        phi = exp(log_phi)
        log_P0_ratio = log1p(-P0) - log(P0)
        log_EC50g = gamma_hill * log(EC50)

        # ── Priors ──
        target += normal_lpdf(log_shape, 0.2, 0.5)
        target += normal_lpdf(log_scale, 2.5, 1.0)
        target += multi_normal_diag_lpdf(beta_s, 0.0, 2.0)
        target += multi_normal_diag_lpdf(beta_k, 0.0, 0.5)
        target += cauchy_lpdf(sigma_country_k, 0.0, 0.5)
        target += cauchy_lpdf(sigma_country_s, 0.0, 1.0)
        target += multi_normal_diag_lpdf(mu_country_k, 0.0, sigma_country_k)
        target += multi_normal_diag_lpdf(mu_country_s, 0.0, sigma_country_s)
        target += normal_lpdf(mu_k, log(0.08), 0.5)
        target += cauchy_lpdf(omega_k, 0.0, 0.5)
        target += normal_lpdf(gamma_k, 1.0, 0.5)
        target += normal_lpdf(gamma_hill, 3.0, 1.0)
        target += beta_lpdf(EC50, 4.0, 6.0)
        target += normal_lpdf(log_phi, log(15.0), 0.5)
        target += beta_lpdf(P0, 2.0, 8.0)
        target += multi_normal_diag_lpdf(z_k, 0.0, 1.0)

        # ── Tier 2: log_k_2, log_eff_scale_2 ──
        @for begin
            log_k_2 = mu_k .+ (tier2_X * beta_k) .+ mu_country_k[tier2_country_ids] .+ (omega_k .* z_k)
            log_eff_scale_2 = log_scale .- ((tier2_X * beta_s) .+ mu_country_s[tier2_country_ids] .+ gamma_k .* log_k_2) .* inv_shape
        end

        for idx in tier2_obs_idx
            target += weibull_logsigma_lpdf(tier2_times[idx], shape, log_eff_scale_2[idx])
        end
        for idx in tier2_cens_idx
            target += weibull_logsigma_lccdf(tier2_times[idx], shape, log_eff_scale_2[idx])
        end

        # ── Tier 1: log_eff_scale_1 ──
        @for begin
            log_k_1 = mu_k .+ (tier1_X * beta_k) .+ mu_country_k[tier1_country_ids]
            log_eff_scale_1 = log_scale .- ((tier1_X * beta_s) .+ mu_country_s[tier1_country_ids] .+ gamma_k .* log_k_1) .* inv_shape
        end

        for idx in tier1_obs_idx
            target += weibull_logsigma_lpdf(tier1_times[idx], shape, log_eff_scale_1[idx])
        end
        for idx in tier1_cens_idx
            target += weibull_logsigma_lccdf(tier1_times[idx], shape, log_eff_scale_1[idx])
        end

        # ── MRC likelihood (scalar loop, zero extra alloc) ──
        for i in 1:total_mrc_obs
            k_i = exp(log_k_2[mrc_patient_ids[i]])
            P_t_i = 1.0 / (1.0 + exp(log_P0_ratio - k_i * mrc_times_flat[i]))
            log_Pg_i = gamma_hill * log(max(P_t_i, 1e-9))
            mu_i = clamp(1.0 / (1.0 + exp(log_Pg_i - log_EC50g)), 1e-6, 1.0 - 1e-6)
            a_mrc = mu_i * phi
            b_mrc = (1.0 - mu_i) * phi
            target += beta_binomial_lpdf(mrc_scores_flat[i], MRC_MAX, a_mrc, b_mrc)
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Ground truth parameters
# ─────────────────────────────────────────────────────────────────────────────

Random.seed!(42)

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
# Simulate data from the generative model
# ─────────────────────────────────────────────────────────────────────────────

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
# Instantiate & sample
# ─────────────────────────────────────────────────────────────────────────────

d = JointALM_DataSet(
    n1=n1, n2=n2, p=p, n_countries=n_countries, MRC_MAX=MRC_MAX,
    tier1_times=tier1_times, tier1_X=tier1_X,
    tier1_country_ids=tier1_country_ids,
    tier1_obs_idx=tier1_obs_idx, tier1_cens_idx=tier1_cens_idx,
    tier2_times=tier2_times, tier2_X=tier2_X,
    tier2_country_ids=tier2_country_ids,
    tier2_obs_idx=tier2_obs_idx, tier2_cens_idx=tier2_cens_idx,
    total_mrc_obs=total_mrc_obs, mrc_scores_flat=mrc_scores_flat,
    mrc_times_flat=mrc_times_flat, mrc_patient_ids=mrc_patient_ids
)

m = make_jointalm(d)
println("\nJoint ALM Model — dim=$(m.dim)")

q_test = randn(m.dim)
lp = log_prob(m, q_test)
println("Test log_prob = $(round(lp; sigdigits=4))")
@assert isfinite(lp) "log_prob is not finite at test point"

println("\nSampling 10000 draws (500 warmup)...")
@time samples = Skate.sample(m, 2000; ϵ=0.1, max_depth=8, warmup=1000,chains=4);
println("Done — $(length(samples)) draws\n")