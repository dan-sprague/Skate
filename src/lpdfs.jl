using SpecialFunctions: loggamma, logbeta
using LinearAlgebra: dot, diag
const _LOG_ZERO = -Inf

"""
    multi_normal_cholesky_lpdf(x, μ, L)

Stan-style log-density for MVN.
L is the Lower-Triangular Cholesky factor of the covariance matrix.
"""
function multi_normal_cholesky_lpdf(x, μ, L)
    k = length(x)

    # Forward substitution: solve L*z = (x - μ), accumulate z'z and log|det L|
    z = Vector{Float64}(undef, k)
    log_det = 0.0
    @inbounds for i in 1:k
        L[i, i] <= 0 && return _LOG_ZERO
        s = x[i] - μ[i]
        for j in 1:i-1
            s -= L[i, j] * z[j]
        end
        z[i] = s / L[i, i]
        log_det += log(L[i, i])
    end

    quad = 0.0
    @inbounds for i in 1:k
        quad += z[i] * z[i]
    end

    return -0.5 * quad - log_det - 0.5 * k * log(2π)
end

# Scalar μ, σ — BLAS dot/sum, zero allocations.
# ||x - μ·1||² = x·x - 2μ·Σxᵢ + n·μ²
function multi_normal_diag_lpdf(x, μ::Real, σ::Real)
    σ <= 0 && return _LOG_ZERO
    σ_safe = max(σ, eps(Float64))
    n = length(x)
    ss = dot(x, x) - 2μ * sum(x) + n * μ^2
    return -0.5 * ss / (σ_safe * σ_safe) - n * log(σ_safe) - 0.5 * n * log(2π)
end

# Vector μ, scalar σ — element-wise loop, zero allocations.
function multi_normal_diag_lpdf(x, μ::AbstractVector, σ::Real)
    σ <= 0 && return _LOG_ZERO
    σ_safe = max(σ, eps(Float64))
    n = length(x)
    inv_σ2 = 1.0 / (σ_safe * σ_safe)
    ss = 0.0
    @inbounds for i in eachindex(x, μ)
        d = x[i] - μ[i]
        ss += d * d
    end
    return -0.5 * ss * inv_σ2 - n * log(σ_safe) - 0.5 * n * log(2π)
end

# Vector μ, vector σ — element-wise loop, zero allocations.
function multi_normal_diag_lpdf(x, μ::AbstractVector, σ::AbstractVector)
    n = length(x)
    lp = -0.5 * n * log(2π)
    @inbounds for i in eachindex(x, μ, σ)
        σi = σ[i]
        σi <= 0 && return _LOG_ZERO
        σ_safe = max(σi, eps(Float64))
        z = (x[i] - μ[i]) / σ_safe
        lp -= 0.5 * z * z + log(σ_safe)
    end
    return lp
end

"""Construct the Cholesky factor of a covariance matrix from scales and correlation Cholesky.
Equivalent to `Diagonal(sigma) * L_corr`, i.e. `sigma .* L_corr`."""
diag_pre_multiply(sigma, L) = sigma .* L

"""
    multi_normal_cholesky_scaled_lpdf(x, μ, log_sigma_row, L_corr)

Fused MVN Cholesky log-density with log-scale parameterization.
Equivalent to `multi_normal_cholesky_lpdf(x, μ, diag_pre_multiply(exp.(log_sigma), L_corr))`
but computes `exp()` per-element inline — no broadcast allocation, no `diag_pre_multiply` temp.
"""
function multi_normal_cholesky_scaled_lpdf(x, μ, log_sigma_row, L_corr)
    k = length(x)
    z = Vector{Float64}(undef, k)
    log_det = 0.0
    @inbounds for i in 1:k
        σi = exp(log_sigma_row[i])
        Lii = σi * L_corr[i, i]
        Lii <= 0 && return _LOG_ZERO
        s = x[i] - μ[i]
        for j in 1:i-1
            s -= σi * L_corr[i, j] * z[j]
        end
        z[i] = s / Lii
        log_det += log(Lii)
    end
    quad = 0.0
    @inbounds for i in 1:k
        quad += z[i] * z[i]
    end
    return -0.5 * quad - log_det - 0.5 * k * log(2π)
end

function normal_lpdf(x, μ, σ)
    if σ <= 0
        return _LOG_ZERO
    end
    σ_safe = max(σ, eps(typeof(float(σ))))
    return -log(σ_safe) - 0.5 * log(2π) - 0.5 * abs2((x - μ) / σ_safe)
end

function cauchy_lpdf(x, μ, γ)
    if γ <= 0
        return _LOG_ZERO
    end
    γ_safe = max(γ, eps(typeof(float(γ))))
    return -log(π) + log(γ_safe) - log(abs2(x - μ) + abs2(γ_safe))
end

function exponential_lpdf(x, λ)
    if λ <= 0 || x < 0
        return _LOG_ZERO
    end
    λ_safe = max(λ, eps(typeof(float(λ))))
    return -log(λ_safe) - x / λ_safe
end

function gamma_lpdf(x, α, β)
    if x <= 0 || α <= 0 || β <= 0
        return _LOG_ZERO
    end
    x_safe = max(x, eps(typeof(float(x))))
    α_safe = max(α, eps(typeof(float(α))))
    β_safe = max(β, eps(typeof(float(β))))
    return α_safe * log(β_safe) - loggamma(α_safe) + (α_safe - 1) * log(x_safe) - β_safe * x_safe
end

function poisson_lpdf(x, λ)
    if λ <= 0 || x < 0
        return _LOG_ZERO
    end
    λ_safe = max(λ, eps(typeof(float(λ))))
    return x * log(λ_safe) - λ_safe - loggamma(x + 1)
end

function binomial_lpdf(x, n, p)
    if p < 0 || p > 1 || x < 0 || x > n
        return _LOG_ZERO
    end
    p_safe = clamp(p, eps(typeof(float(p))), one(p) - eps(typeof(float(p))))
    log_n_choose_x = -log(n + 1) - logbeta(n - x + 1, x + 1)
    return log_n_choose_x + x * log(p_safe) + (n - x) * log(1 - p_safe)
end

function beta_binomial_lpdf(x, n, α, β)
    if α <= 0 || β <= 0 || x < 0 || x > n
        return _LOG_ZERO
    end
    α_safe = max(α, eps(typeof(float(α))))
    β_safe = max(β, eps(typeof(float(β))))
    return -log(n + 1) - logbeta(n - x + 1, x + 1) + logbeta(x + α_safe, n - x + β_safe) - logbeta(α_safe, β_safe)
end

function weibull_lpdf(x, α, σ)
    if x < 0 || α <= 0 || σ <= 0
        return _LOG_ZERO
    end
    x_safe = max(x, eps(typeof(float(x))))
    σ_safe = max(σ, eps(typeof(float(σ))))
    α_safe = max(α, eps(typeof(float(α))))
    return log(α_safe) - log(σ_safe) + (α_safe - 1) * (log(x_safe) - log(σ_safe)) - (x_safe / σ_safe)^α_safe
end

function weibull_lccdf(x, α, σ)
    if x < 0 || α <= 0 || σ <= 0
        return _LOG_ZERO
    end
    x_safe = max(x, zero(x))
    σ_safe = max(σ, eps(typeof(float(σ))))
    α_safe = max(α, eps(typeof(float(α))))
    return -(x_safe / σ_safe)^α_safe
end

"""
    neg_binomial_2_lpdf(y, μ, ϕ)

Stan-style Negative Binomial log-density.
μ: Mean
ϕ: Dispersion (smaller ϕ = more variance/overdispersion)
"""
function neg_binomial_2_lpdf(y, μ, ϕ)
    if y < 0 || μ <= 0 || ϕ <= 0
        return _LOG_ZERO
    end
    μ_safe = max(μ, eps(typeof(float(μ))))
    ϕ_safe = max(ϕ, eps(typeof(float(ϕ))))
    
    term1 = loggamma(y + ϕ_safe) - loggamma(y + 1) - loggamma(ϕ_safe)
    term2 = ϕ_safe * (log(ϕ_safe) - log(ϕ_safe + μ_safe))
    term3 = y * (log(μ_safe) - log(ϕ_safe + μ_safe))

    return term1 + term2 + term3
end

"""
    bernoulli_logit_lpdf(y, α)

Stan-style Bernoulli log-density using the logit-link linear predictor α.
α is typically (intercept + X * beta).
"""
function bernoulli_logit_lpdf(y, α)
    if y != 0 && y != 1
        return _LOG_ZERO
    end
    return y * α - (log1p(exp(-abs(α))) + max(zero(α), α))
end

"""
    binomial_logit_lpdf(y, n, α)
"""
function binomial_logit_lpdf(y, n, α)
    if y < 0 || y > n
        return _LOG_ZERO
    end
    log_n_choose_y = -log(n + 1) - logbeta(n - y + 1, y + 1)
    return log_n_choose_y + y * α - n * (log1p(exp(-abs(α))) + max(zero(α), α))
end

function weibull_logsigma_lpdf(x, α, log_σ)
    if x < 0 || α <= 0
        return _LOG_ZERO
    end
    x_safe = max(x, eps(typeof(float(x))))
    α_safe = max(α, eps(typeof(float(α))))
    return log(α_safe) - log_σ + (α_safe - 1) * (log(x_safe) - log_σ) - exp(α_safe * (log(x_safe) - log_σ))
end

function categorical_logit_lpdf(y, α_vec)
    if y < 1 || y > length(α_vec) || !isinteger(y)
        return _LOG_ZERO
    end
    return α_vec[Int(y)] - log_sum_exp(α_vec)
end

"""
    weibull_logsigma_lccdf(x, α, log_σ)
"""
function weibull_logsigma_lccdf(x, α, log_σ)
    if x < 0 || α <= 0
        return _LOG_ZERO
    end
    α_safe = max(α, eps(typeof(float(α))))
    x_safe = max(x, zero(x))
    return -exp(α_safe * (log(x_safe) - log_σ))
end

"""
    lkj_corr_cholesky_lpdf(L, η)
"""
function lkj_corr_cholesky_lpdf(L, η)
    if η <= 0
        return _LOG_ZERO
    end
    
    K = size(L, 1)
    s = 0.0
    for i in 1:K
        if L[i, i] <= 0
            return _LOG_ZERO
        end
        s += (K - i + 2*(η - 1)) * log(L[i, i])
    end
    return s
end

# Beta: logpdf(Beta(α, β), x)
function beta_lpdf(x, α, β)
    if x <= 0 || x >= 1 || α <= 0 || β <= 0
        return _LOG_ZERO
    end
    x_safe = clamp(x, eps(typeof(float(x))), one(x) - eps(typeof(float(x))))
    α_safe = max(α, eps(typeof(float(α))))
    β_safe = max(β, eps(typeof(float(β))))
    return (α_safe - 1)*log(x_safe) + (β_safe - 1)*log(1 - x_safe) - logbeta(α_safe, β_safe)
end

function lognormal_lpdf(x, μ, σ)
    if x <= 0 || σ <= 0
        return _LOG_ZERO
    end
    x_safe = max(x, eps(typeof(float(x))))
    σ_safe = max(σ, eps(typeof(float(σ))))
    return -log(x_safe) - log(σ_safe) - 0.5*log(2π) - 0.5*((log(x_safe) - μ)/σ_safe)^2
end

function student_t_lpdf(x, ν, μ, σ)
    if ν <= 0 || σ <= 0
        return _LOG_ZERO
    end
    σ_safe = max(σ, eps(typeof(float(σ))))
    ν_safe = max(ν, eps(typeof(float(ν))))
    z = (x - μ) / σ_safe
    return loggamma(0.5*(ν_safe + 1)) - loggamma(0.5*ν_safe) -
           0.5*log(ν_safe*π) - log(σ_safe) -
           0.5*(ν_safe + 1)*log(1 + z^2/ν_safe)
end

function dirichlet_lpdf(x, α)
    K = length(x)
    length(α) == K || error("dirichlet_lpdf: x and α must have the same length")
    eps_val = eps(Float64)
    sum_α = 0.0
    sum_loggamma_α = 0.0
    kernel = 0.0
    for i in 1:K
        xi = x[i]; αi = α[i]
        (xi <= 0 || αi <= 0) && return _LOG_ZERO
        xi_safe = clamp(xi, eps_val, 1.0 - eps_val)
        αi_safe = max(αi, eps_val)
        sum_α += αi_safe
        sum_loggamma_α += loggamma(αi_safe)
        kernel += (αi_safe - 1) * log(xi_safe)
    end
    return loggamma(sum_α) - sum_loggamma_α + kernel
end

"""
    dirichlet_lpdf(x, K::Float64)

Symmetric Dirichlet with α = 1 (uniform over the simplex).
K is the dimension — equivalent to `dirichlet_lpdf(x, ones(K))` but zero-allocation.
"""
function dirichlet_lpdf(x, α::Float64)
    α > 0 || error("dirichlet_lpdf: α must be positive")
    K = length(x)
    eps_val = eps(Float64)
    α_safe = max(α, eps_val)
    kernel = 0.0
    for i in 1:K
        x[i] <= 0 && return _LOG_ZERO
        kernel += (α_safe - 1) * log(clamp(x[i], eps_val, 1.0 - eps_val))
    end
    return loggamma(K * α_safe) - K * loggamma(α_safe) + kernel
end

function uniform_lpdf(x, lo, hi)
    if x < lo || x > hi || lo >= hi
        return _LOG_ZERO
    end
    return -log(hi - lo)
end

function laplace_lpdf(x, μ, b)
    if b <= 0
        return _LOG_ZERO
    end
    b_safe = max(b, eps(typeof(float(b))))
    return -log(2*b_safe) - abs(x - μ)/b_safe
end

function logistic_lpdf(x, μ, s)
    if s <= 0
        return _LOG_ZERO
    end
    s_safe = max(s, eps(typeof(float(s))))
    z = (x - μ) / s_safe
    return -log(s_safe) - z - 2*log1p(exp(-z))
end