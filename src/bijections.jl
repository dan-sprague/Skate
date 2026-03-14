abstract type Constraint end

"""No-op constraint for unconstrained parameters."""
struct IdentityConstraint <: Constraint end
transform(::IdentityConstraint, x) = x
log_abs_det_jacobian(::IdentityConstraint, x) = zero(typeof(x))
grad_correction(::IdentityConstraint, x) = one(typeof(x))

# ── ExpBijection family: shared Jacobian and grad_correction ──────────────────
abstract type ExpBijection <: Constraint end
log_abs_det_jacobian(::ExpBijection, x) = x
grad_correction(::ExpBijection, x) = exp(x)

"""Exp transform enforcing `x > lb`."""
struct LowerBounded{T} <: ExpBijection lb::T end
LowerBounded(lb::Real) = LowerBounded{Float64}(Float64(lb))
transform(c::LowerBounded, x) = c.lb + exp(x)

"""Exp transform enforcing `x < ub`."""
struct UpperBounded{T} <: ExpBijection ub::T end
UpperBounded(ub::Real) = UpperBounded{Float64}(Float64(ub))
transform(c::UpperBounded, x) = c.ub - exp(x)

"""Logistic-sigmoid transform enforcing `lb < x < ub`."""
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

"""Stick-breaking transform mapping K-1 reals to a K-simplex."""
struct SimplexConstraint <: Constraint end

"""
    simplex_transform(y) → (x, log_jac)

Apply stick-breaking simplex transform, returning `(x, log_jac)`.
"""
function simplex_transform(y::AbstractVector{<:Real})
    Km1 = length(y)
    K = Km1 + 1
    x = similar(y, K)
    log_jac = simplex_transform!(x, y)
    return x, log_jac
end

"""
    simplex_transform!(x, y) → log_jac

In-place stick-breaking simplex transform. Returns log Jacobian determinant.
"""
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

"""Cumulative-exp transform producing ordered reals."""
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

"""
    ordered_transform(x) → (y, log_jac)

Apply ordered transform, returning `(y, log_jac)`.
"""
function ordered_transform(x::AbstractVector{<:Real})
    K = length(x)
    y = similar(x)
    log_jac = ordered_transform!(y, x)
    return y, log_jac
end

"""
    ordered_transform!(y, x) → log_jac

In-place ordered transform. Returns log Jacobian determinant.
"""
function ordered_transform!(y::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    y[1] = x[1]
    log_jac = zero(eltype(x))
    for i in 2:length(x)
        y[i] = y[i-1] + exp(x[i])
        log_jac += x[i]
    end
    return log_jac
end

# ── Ordered-row simplex matrix ────────────────────────────────────────────────
# Maps K*(V-1) unconstrained reals → K×V matrix where each row is a V-simplex
# and column 1 is strictly increasing across rows.
#
# Layout in q:
#   q[1:K]                     — first simplex param per row (ordered via cumulative exp)
#   q[K+1 : K+(V-2)]          — remaining V-2 simplex params for row 1
#   q[K+(V-2)+1 : K+2*(V-2)]  — remaining V-2 simplex params for row 2
#   ...
#
# Since phi[k,1] = logistic(y[k,1] - log(V-1)) and logistic is monotone,
# ordering y[k,1] across rows guarantees phi[1,1] < phi[2,1] < ... < phi[K,1].
# Total free params: K + K*(V-2) = K*(V-1), same as pure row simplex.

"""
    ordered_simplex_matrix!(mat, q, K, V) → log_jac

In-place ordered-row simplex matrix transform. Returns log Jacobian determinant.
"""
function ordered_simplex_matrix!(mat::AbstractMatrix{<:Real},
                                 q::AbstractVector{<:Real},
                                 K::Int, V::Int)
    Vm1 = V - 1
    Vm2 = V - 2
    T = eltype(q)
    log_jac = zero(T)
    prev_ordered = q[1]

    for k in 1:K
        # ordered transform for the first simplex param of row k
        if k == 1
            y1 = q[1]
        else
            y1 = prev_ordered + exp(q[k])
            log_jac += q[k]
        end
        prev_ordered = y1

        # stick-breaking simplex transform for row k
        remaining = one(T)

        # first simplex element (uses ordered y1)
        z = _logistic(y1 - log(T(Vm1)))
        mat[k, 1] = z * remaining
        log_jac += log(z) + log(one(T) - z) + log(remaining)
        remaining -= mat[k, 1]

        # remaining V-2 simplex elements
        rest_base = K + (k - 1) * Vm2
        for j in 1:Vm2
            yj = q[rest_base + j]
            z = _logistic(yj - log(T(Vm1 - j)))
            mat[k, j + 1] = z * remaining
            log_jac += log(z) + log(one(T) - z) + log(remaining)
            remaining -= mat[k, j + 1]
        end

        # last simplex element
        mat[k, V] = remaining
    end

    return log_jac
end

"""
    corr_cholesky_transform(z, D) → (L, log_jac)

Canonical partial correlations transform to correlation Cholesky factor. Returns `(L, log_jac)`.
"""
function corr_cholesky_transform(z::AbstractVector{<:Real}, D::Int)
    T = eltype(z)
    L = zeros(T, D, D)
    log_jac = corr_cholesky_transform!(L, z, D)
    return L, log_jac
end

"""
    corr_cholesky_transform!(L, z, D) → log_jac

In-place CPC transform to correlation Cholesky factor. Returns log Jacobian determinant.
"""
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
