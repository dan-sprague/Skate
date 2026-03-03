# @skate model translations of posteriordb Stan models.
# Each model has:
#   1. @skate definition (generates ModelData struct + make function)
#   2. build_*(data_dict) — convert posteriordb JSON data to ModelLogDensity
#   3. extract_*_draws(ch) — extract PhaseSkate Chains into Dict{String, Vector{Float64}}

# ─── Helpers ─────────────────────────────────────────────────────────────────

function _to_float_vec(x)
    Float64.(x)
end

function _to_float_mat(x::AbstractVector)
    nrows = length(x)
    ncols = length(x[1])
    M = Matrix{Float64}(undef, nrows, ncols)
    for i in 1:nrows
        for j in 1:ncols
            M[i, j] = Float64(x[i][j])
        end
    end
    return M
end

# ═════════════════════════════════════════════════════════════════════════════
# 1. Eight Schools (Non-Centered)
# ═════════════════════════════════════════════════════════════════════════════

@skate EightSchools begin
    @constants begin
        J::Int
        y::Vector{Float64}
        sigma::Vector{Float64}
    end
    @params begin
        mu::Float64
        tau = param(Float64; lower=0.0)
        theta_trans = param(Vector{Float64}, J)
    end
    @logjoint begin
        target += normal_lpdf(mu, 0.0, 5.0)
        target += cauchy_lpdf(tau, 0.0, 5.0)
        for j in 1:J
            target += normal_lpdf(theta_trans[j], 0.0, 1.0)
            target += normal_lpdf(y[j], mu + tau * theta_trans[j], sigma[j])
        end
    end
end

function build_eight_schools(data_dict)
    d = EightSchoolsData(
        J     = Int(data_dict["J"]),
        y     = _to_float_vec(data_dict["y"]),
        sigma = _to_float_vec(data_dict["sigma"]),
    )
    return make(d)
end

function extract_eight_schools_draws(ch::Chains)
    draws = Dict{String, Vector{Float64}}()
    draws["mu"]  = vec(samples(ch, :mu))
    draws["tau"] = vec(samples(ch, :tau))
    J = size(samples(ch, :theta_trans), 2)
    for j in 1:J
        draws["theta_trans[$j]"] = vec(samples(ch, :theta_trans)[:, j, :])
    end
    # Derived: theta[j] = mu + tau * theta_trans[j]
    for j in 1:J
        draws["theta[$j]"] = draws["mu"] .+ draws["tau"] .* draws["theta_trans[$j]"]
    end
    return draws
end

# ═════════════════════════════════════════════════════════════════════════════
# 2. Bayesian Linear Regression (sblrc-blr)
# ═════════════════════════════════════════════════════════════════════════════

@skate BLR begin
    @constants begin
        N::Int
        D::Int
        X::Matrix{Float64}
        y::Vector{Float64}
    end
    @params begin
        beta = param(Vector{Float64}, D)
        sigma = param(Float64; lower=0.0)
    end
    @logjoint begin
        for d in 1:D
            target += normal_lpdf(beta[d], 0.0, 10.0)
        end
        target += normal_lpdf(sigma, 0.0, 10.0)
        for i in 1:N
            mu = 0.0
            for d in 1:D
                mu += X[i, d] * beta[d]
            end
            target += normal_lpdf(y[i], mu, sigma)
        end
    end
end

function build_blr(data_dict)
    d = BLRData(
        N = Int(data_dict["N"]),
        D = Int(data_dict["D"]),
        X = _to_float_mat(data_dict["X"]),
        y = _to_float_vec(data_dict["y"]),
    )
    return make(d)
end

function extract_blr_draws(ch::Chains)
    draws = Dict{String, Vector{Float64}}()
    D = size(samples(ch, :beta), 2)
    for d in 1:D
        draws["beta[$d]"] = vec(samples(ch, :beta)[:, d, :])
    end
    draws["sigma"] = vec(samples(ch, :sigma))
    return draws
end

# ═════════════════════════════════════════════════════════════════════════════
# 3. Earnings ~ Height (earnings-earn_height)
# ═════════════════════════════════════════════════════════════════════════════

@skate EarnHeight begin
    @constants begin
        N::Int
        earn::Vector{Float64}
        height::Vector{Float64}
    end
    @params begin
        beta = param(Vector{Float64}, 2)
        sigma = param(Float64; lower=0.0)
    end
    @logjoint begin
        for i in 1:N
            target += normal_lpdf(earn[i], beta[1] + beta[2] * height[i], sigma)
        end
    end
end

function build_earn_height(data_dict)
    d = EarnHeightData(
        N      = Int(data_dict["N"]),
        earn   = _to_float_vec(data_dict["earn"]),
        height = _to_float_vec(data_dict["height"]),
    )
    return make(d)
end

function extract_earn_height_draws(ch::Chains)
    draws = Dict{String, Vector{Float64}}()
    draws["beta[1]"] = vec(samples(ch, :beta)[:, 1, :])
    draws["beta[2]"] = vec(samples(ch, :beta)[:, 2, :])
    draws["sigma"]   = vec(samples(ch, :sigma))
    return draws
end

# ═════════════════════════════════════════════════════════════════════════════
# 4. Kid Score ~ Mom IQ (kidiq-kidscore_momiq)
# ═════════════════════════════════════════════════════════════════════════════

@skate KidScoreMomIQ begin
    @constants begin
        N::Int
        kid_score::Vector{Float64}
        mom_iq::Vector{Float64}
    end
    @params begin
        beta = param(Vector{Float64}, 2)
        sigma = param(Float64; lower=0.0)
    end
    @logjoint begin
        target += cauchy_lpdf(sigma, 0.0, 2.5)
        for i in 1:N
            target += normal_lpdf(kid_score[i], beta[1] + beta[2] * mom_iq[i], sigma)
        end
    end
end

function build_kidscore_momiq(data_dict)
    d = KidScoreMomIQData(
        N         = Int(data_dict["N"]),
        kid_score = _to_float_vec(data_dict["kid_score"]),
        mom_iq    = _to_float_vec(data_dict["mom_iq"]),
    )
    return make(d)
end

function extract_kidscore_momiq_draws(ch::Chains)
    draws = Dict{String, Vector{Float64}}()
    draws["beta[1]"] = vec(samples(ch, :beta)[:, 1, :])
    draws["beta[2]"] = vec(samples(ch, :beta)[:, 2, :])
    draws["sigma"]   = vec(samples(ch, :sigma))
    return draws
end

# ═════════════════════════════════════════════════════════════════════════════
# 5. Autoregressive AR(K) (arK-arK)
# ═════════════════════════════════════════════════════════════════════════════

@skate ARK begin
    @constants begin
        K::Int
        T::Int
        y::Vector{Float64}
    end
    @params begin
        alpha::Float64
        beta = param(Vector{Float64}, K)
        sigma = param(Float64; lower=0.0)
    end
    @logjoint begin
        target += normal_lpdf(alpha, 0.0, 10.0)
        for k in 1:K
            target += normal_lpdf(beta[k], 0.0, 10.0)
        end
        target += cauchy_lpdf(sigma, 0.0, 2.5)
        for t in (K+1):T
            mu = alpha
            for k in 1:K
                mu += beta[k] * y[t - k]
            end
            target += normal_lpdf(y[t], mu, sigma)
        end
    end
end

function build_ark(data_dict)
    d = ARKData(
        K = Int(data_dict["K"]),
        T = Int(data_dict["T"]),
        y = _to_float_vec(data_dict["y"]),
    )
    return make(d)
end

function extract_ark_draws(ch::Chains)
    draws = Dict{String, Vector{Float64}}()
    draws["alpha"] = vec(samples(ch, :alpha))
    K = size(samples(ch, :beta), 2)
    for k in 1:K
        draws["beta[$k]"] = vec(samples(ch, :beta)[:, k, :])
    end
    draws["sigma"] = vec(samples(ch, :sigma))
    return draws
end

# ═════════════════════════════════════════════════════════════════════════════
# 6. Low-Dimensional Gaussian Mixture (low_dim_gauss_mix-low_dim_gauss_mix)
# ═════════════════════════════════════════════════════════════════════════════

@skate LowDimGaussMix begin
    @constants begin
        N::Int
        y::Vector{Float64}
    end
    @params begin
        mu = param(Vector{Float64}, 2; ordered=true)
        sigma1 = param(Float64; lower=0.0)
        sigma2 = param(Float64; lower=0.0)
        theta = param(Float64; lower=0.0, upper=1.0)
    end
    @logjoint begin
        target += normal_lpdf(sigma1, 0.0, 2.0)
        target += normal_lpdf(sigma2, 0.0, 2.0)
        target += normal_lpdf(mu[1], 0.0, 2.0)
        target += normal_lpdf(mu[2], 0.0, 2.0)
        target += beta_lpdf(theta, 5.0, 5.0)
        for n in 1:N
            lp1 = log(theta) + normal_lpdf(y[n], mu[1], sigma1)
            lp2 = log(1.0 - theta) + normal_lpdf(y[n], mu[2], sigma2)
            if lp1 > lp2
                target += lp1 + log1p(exp(lp2 - lp1))
            else
                target += lp2 + log1p(exp(lp1 - lp2))
            end
        end
    end
end

function build_low_dim_gauss_mix(data_dict)
    d = LowDimGaussMixData(
        N = Int(data_dict["N"]),
        y = _to_float_vec(data_dict["y"]),
    )
    return make(d)
end

function extract_low_dim_gauss_mix_draws(ch::Chains)
    draws = Dict{String, Vector{Float64}}()
    draws["mu[1]"]    = vec(samples(ch, :mu)[:, 1, :])
    draws["mu[2]"]    = vec(samples(ch, :mu)[:, 2, :])
    draws["sigma[1]"] = vec(samples(ch, :sigma1))
    draws["sigma[2]"] = vec(samples(ch, :sigma2))
    draws["theta"]    = vec(samples(ch, :theta))
    return draws
end

# ═════════════════════════════════════════════════════════════════════════════
# Model registry
# ═════════════════════════════════════════════════════════════════════════════

struct PDBTestModel
    posterior_name::String
    build::Function
    extract_draws::Function
end

const POSTERIORDB_MODELS = [
    PDBTestModel("eight_schools-eight_schools_noncentered",
                 build_eight_schools, extract_eight_schools_draws),
    PDBTestModel("sblrc-blr",
                 build_blr, extract_blr_draws),
    PDBTestModel("earnings-earn_height",
                 build_earn_height, extract_earn_height_draws),
    PDBTestModel("kidiq-kidscore_momiq",
                 build_kidscore_momiq, extract_kidscore_momiq_draws),
    PDBTestModel("arK-arK",
                 build_ark, extract_ark_draws),
    PDBTestModel("low_dim_gauss_mix-low_dim_gauss_mix",
                 build_low_dim_gauss_mix, extract_low_dim_gauss_mix_draws),
]
