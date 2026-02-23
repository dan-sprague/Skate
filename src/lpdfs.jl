using SpecialFunctions: loggamma, logbeta
const _LOG_ZERO = -Inf 

"""
    multi_normal_cholesky_lpdf(x, μ, L)

Stan-style log-density for MVN.
L is the Lower-Triangular Cholesky factor of the covariance matrix.
"""
function multi_normal_cholesky_lpdf(x, μ, L)
    if any(i -> L[i, i] <= 0, axes(L, 1))
        return _LOG_ZERO
    end

    z = L \ (x .- μ)
    quad_form = -0.5 * dot(z, z)

    # -½ log|Σ|: since Σ = LLᵀ, log|Σ| = 2 Σᵢ log(Lᵢᵢ), so -½ log|Σ| = -Σᵢ log(Lᵢᵢ)
    neg_half_log_det = -sum(log.(diag(L)))

    k = length(x)
    const_term = -0.5 * k * log(2π)

    return quad_form + neg_half_log_det + const_term
end

function multi_normal_diag_lpdf(x, μ, σ)
    if any(s -> s <= 0, σ)
        return _LOG_ZERO
    end

    σ_safe = max.(σ, eps(eltype(float.(σ))))
    z = (x .- μ) ./ σ_safe

    quad_form = -0.5 * sum(abs2, z)
    n = length(x)
    log_det = -n * sum(log.(σ_safe))
    const_term = -0.5 * n * log(2π)

    return quad_form + log_det + const_term
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
    if any(v -> v <= 0, x) || any(v -> v <= 0, α)
        return _LOG_ZERO
    end
    x_safe = clamp.(x, eps(eltype(float.(x))), one(eltype(x)) - eps(eltype(float.(x))))
    α_safe = max.(α, eps(eltype(float.(α))))
    return loggamma(sum(α_safe)) - sum(loggamma.(α_safe)) + sum((α_safe .- 1) .* log.(x_safe))
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