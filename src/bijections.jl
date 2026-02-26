abstract type Constraint end

struct IdentityConstraint <: Constraint end
transform(::IdentityConstraint, x) = x
log_abs_det_jacobian(::IdentityConstraint, x) = zero(typeof(x))
grad_correction(::IdentityConstraint, x) = one(typeof(x))

# ── ExpBijection family: shared Jacobian and grad_correction ──────────────────
abstract type ExpBijection <: Constraint end
log_abs_det_jacobian(::ExpBijection, x) = x
grad_correction(::ExpBijection, x) = exp(x)

struct LowerBounded{T} <: ExpBijection lb::T end
LowerBounded(lb::Real) = LowerBounded{Float64}(Float64(lb))
transform(c::LowerBounded, x) = c.lb + exp(x)

struct UpperBounded{T} <: ExpBijection ub::T end
UpperBounded(ub::Real) = UpperBounded{Float64}(Float64(ub))
transform(c::UpperBounded, x) = c.ub - exp(x)

# ── Bounded (logistic sigmoid) ────────────────────────────────────────────────
struct Bounded{T} <: Constraint
    lb::T
    ub::T
    function Bounded{T}(lb, ub) where T
        lb >= ub && error("Lower bound must be less than upper bound")
        new{T}(T(lb), T(ub))
    end
end
Bounded(lb::T, ub::T) where {T<:Real} = Bounded{T}(lb, ub)
Bounded(lb, ub) = Bounded{Float64}(Float64(lb), Float64(ub))

# Branchless logistic — IEEE-754 safe: exp(-large_neg)=Inf, inv(1+Inf)=0.0
_logistic(x) = inv(one(x) + exp(-x))

function transform(c::Bounded, x)
    s = _logistic(x)
    c.lb + (c.ub - c.lb) * s
end

function log_abs_det_jacobian(c::Bounded, x)
    s = _logistic(x)
    log(c.ub - c.lb) + log(s) + log(one(s) - s)
end

function grad_correction(c::Bounded, x)
    s = _logistic(x)
    (c.ub - c.lb) * s * (one(s) - s)
end

# ── Simplex (stick-breaking transform) ───────────────────────────────────────
# Maps K-1 unconstrained reals → K-simplex.
# y_k = z_k * remaining,  where z_k = logistic(x_k - log(K - k))
# The centering adjustment -log(K-k) makes the uniform prior map to the
# center of the simplex.
struct SimplexConstraint <: Constraint end

function simplex_transform(y::AbstractVector{<:Real})
    Km1 = length(y)
    K = Km1 + 1
    x = similar(y, K)
    log_jac = simplex_transform!(x, y)
    return x, log_jac
end

function simplex_transform!(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    Km1 = length(y)
    K = Km1 + 1
    T = eltype(y)
    log_jac = zero(T)
    remaining = one(T)

    for k in 1:Km1
        z = _logistic(y[k] - log(T(K - k)))
        x[k] = z * remaining
        log_jac += log(z) + log(one(T) - z) + log(remaining)
        remaining -= x[k]
    end
    x[K] = remaining
    return log_jac
end

# ── Ordered (cumulative exp transform) ───────────────────────────────────────
# Maps K unconstrained reals → K ordered reals: y₁ = x₁, yₖ = yₖ₋₁ + exp(xₖ)
struct OrderedConstraint <: Constraint end

function transform(::OrderedConstraint, x::AbstractVector{<:Real})
    y = similar(x)
    y[1] = x[1]
    for i in 2:lastindex(x)
        y[i] = y[i-1] + exp(x[i])
    end
    return y
end

function log_abs_det_jacobian(::OrderedConstraint, x::AbstractVector{<:Real})
    sum(@view x[2:end])
end

function ordered_transform(x::AbstractVector{<:Real})
    K = length(x)
    y = similar(x)
    log_jac = ordered_transform!(y, x)
    return y, log_jac
end

function ordered_transform!(y::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    y[1] = x[1]
    log_jac = zero(eltype(x))
    for i in 2:length(x)
        y[i] = y[i-1] + exp(x[i])
        log_jac += x[i]
    end
    return log_jac
end

# ── Correlation Cholesky (CPC / tanh parameterization) ───────────────────────
# Maps D*(D-1)/2 unconstrained reals → D×D lower-triangular Cholesky factor
# of a correlation matrix (rows have unit norm, so LL^T has ones on diagonal).
# Uses Canonical Partial Correlations: each z maps through tanh to (-1,1).

function corr_cholesky_transform(z::AbstractVector{<:Real}, D::Int)
    T = eltype(z)
    L = zeros(T, D, D)
    log_jac = corr_cholesky_transform!(L, z, D)
    return L, log_jac
end

function corr_cholesky_transform!(L::AbstractMatrix{<:Real}, z::AbstractVector{<:Real}, D::Int)
    T = promote_type(eltype(L), eltype(z))
    log_jac = zero(T)
    L[1, 1] = one(T)
    pos = 1
    for i in 2:D
        acc = one(T)
        for j in 1:(i-1)
            w = tanh(z[pos])
            L[i, j] = w * sqrt(acc)
            w2 = w * w
            log_jac += log(one(T) - w2)       # tanh Jacobian: d(tanh)/dz = 1 - tanh²
            log_jac += T(0.5) * log(acc)       # scaling from sqrt(remaining)
            acc *= (one(T) - w2)
            pos += 1
        end
        L[i, i] = sqrt(max(acc, zero(T)))
    end
    return log_jac
end
