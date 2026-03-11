## Custom Enzyme adjoint rules for lpdf functions.
## Zero-allocation: all rules use nothing tape and recompute in reverse.
##
## NOTE: Rules for Duplicated-argument functions (multi_normal_diag_lpdf, etc.)
## are NOT safe with Enzyme + Julia parallel GC. Only scalar Active/Const rules here.

import .Enzyme.EnzymeRules
using .Enzyme: Const, Active, Duplicated

# ═══════════════════════════════════════════════════════════════════════════════
# normal_lpdf — all activity combinations for scalar args
# ∂f/∂x = -(x-μ)/σ²,  ∂f/∂μ = (x-μ)/σ²,  ∂f/∂σ = -1/σ + (x-μ)²/σ³
# ═══════════════════════════════════════════════════════════════════════════════

for (xT, μT, σT) in [
    (Active, Const, Const), (Const, Active, Active), (Active, Active, Active),
    (Active, Const, Active), (Const, Active, Const), (Active, Active, Const),
    (Const, Const, Active)]

    @eval function EnzymeRules.augmented_primal(config, ::Const{typeof(normal_lpdf)},
            ::Type{<:Active}, x::$xT, μ::$μT, σ::$σT)
        σv = max(σ.val, eps(Float64))
        z = (x.val - μ.val) / σv
        lp = -log(σv) - 0.5 * log(2π) - 0.5 * z * z
        return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
    end
    @eval function EnzymeRules.reverse(config, ::Const{typeof(normal_lpdf)},
            dret::Active, tape, x::$xT, μ::$μT, σ::$σT)
        σv = max(σ.val, eps(Float64))
        z = (x.val - μ.val) / σv
        dr = dret.val
        dx = $xT === Active ? dr * (-z / σv) : nothing
        dμ = $μT === Active ? dr * (z / σv) : nothing
        dσ = $σT === Active ? dr * (-1.0/σv + z*z/σv) : nothing
        return (dx, dμ, dσ)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# exponential_lpdf(x, λ)  — x Active, λ Const
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(exponential_lpdf)},
        ::Type{<:Active}, x::Active, λ::Const)
    λv = max(λ.val, eps(Float64))
    lp = -log(λv) - x.val / λv
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(exponential_lpdf)},
        dret::Active, tape, x::Active, λ::Const)
    λv = max(λ.val, eps(Float64))
    return (dret.val * (-1.0 / λv), nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# cauchy_lpdf(x, μ, γ)  — x Active, μ,γ Const
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(cauchy_lpdf)},
        ::Type{<:Active}, x::Active, μ::Const, γ::Const)
    γv = max(γ.val, eps(Float64))
    d = x.val - μ.val
    denom = d*d + γv*γv
    lp = -log(π) + log(γv) - log(denom)
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(cauchy_lpdf)},
        dret::Active, tape, x::Active, μ::Const, γ::Const)
    γv = max(γ.val, eps(Float64))
    d = x.val - μ.val
    denom = d*d + γv*γv
    return (dret.val * (-2.0 * d / denom), nothing, nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# bernoulli_logit_lpdf(y, α)  — y Const, α Active
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(bernoulli_logit_lpdf)},
        ::Type{<:Active}, y::Const, α::Active)
    αv = α.val
    lp = y.val * αv - (log1p(exp(-abs(αv))) + max(0.0, αv))
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(bernoulli_logit_lpdf)},
        dret::Active, tape, y::Const, α::Active)
    sigmoid = 1.0 / (1.0 + exp(-α.val))
    return (nothing, dret.val * (Float64(y.val) - sigmoid))
end

# ═══════════════════════════════════════════════════════════════════════════════
# neg_binomial_2_lpdf(y, μ, ϕ)  — y Const, μ Active, ϕ Const
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(neg_binomial_2_lpdf)},
        ::Type{<:Active}, y::Const, μ::Active, ϕ::Const)
    yv = Float64(y.val); μv = max(μ.val, eps(Float64)); ϕv = max(ϕ.val, eps(Float64))
    lp = _loggamma(yv + ϕv) - _loggamma(yv + 1) - _loggamma(ϕv) +
         ϕv * (log(ϕv) - log(ϕv + μv)) + yv * (log(μv) - log(ϕv + μv))
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(neg_binomial_2_lpdf)},
        dret::Active, tape, y::Const, μ::Active, ϕ::Const)
    yv = Float64(y.val); μv = max(μ.val, eps(Float64)); ϕv = max(ϕ.val, eps(Float64))
    dμ = yv / μv - (yv + ϕv) / (ϕv + μv)
    return (nothing, dret.val * dμ, nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# weibull_logsigma_lpdf(x, α, log_σ)  — all Active
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(weibull_logsigma_lpdf)},
        ::Type{<:Active}, x::Active, α::Active, log_σ::Active)
    xv = max(x.val, eps(Float64)); αv = max(α.val, eps(Float64)); ls = log_σ.val
    log_x = log(xv); diff = log_x - ls
    eterm = exp(αv * diff)
    lp = log(αv) - ls + (αv - 1.0) * diff - eterm
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(weibull_logsigma_lpdf)},
        dret::Active, tape, x::Active, α::Active, log_σ::Active)
    xv = max(x.val, eps(Float64)); αv = max(α.val, eps(Float64)); ls = log_σ.val
    diff = log(xv) - ls; eterm = exp(αv * diff)
    dr = dret.val
    return (dr * ((αv - 1.0) / xv - αv * eterm / xv),
            dr * (1.0/αv + diff - diff * eterm),
            dr * (-αv + αv * eterm))
end

# ═══════════════════════════════════════════════════════════════════════════════
# weibull_logsigma_lccdf(x, α, log_σ)  — all Active
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(weibull_logsigma_lccdf)},
        ::Type{<:Active}, x::Active, α::Active, log_σ::Active)
    xv = max(x.val, eps(Float64)); αv = max(α.val, eps(Float64)); ls = log_σ.val
    diff = log(xv) - ls; eterm = exp(αv * diff)
    lp = -eterm
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(weibull_logsigma_lccdf)},
        dret::Active, tape, x::Active, α::Active, log_σ::Active)
    xv = max(x.val, eps(Float64)); αv = max(α.val, eps(Float64)); ls = log_σ.val
    diff = log(xv) - ls; eterm = exp(αv * diff)
    dr = dret.val
    return (dr * (-αv * eterm / xv), dr * (-diff * eterm), dr * (αv * eterm))
end

# ═══════════════════════════════════════════════════════════════════════════════
# laplace_lpdf(x, μ, b)  — x Active, μ,b Const
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(laplace_lpdf)},
        ::Type{<:Active}, x::Active, μ::Const, b::Const)
    bv = max(b.val, eps(Float64))
    lp = -log(2*bv) - abs(x.val - μ.val)/bv
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(laplace_lpdf)},
        dret::Active, tape, x::Active, μ::Const, b::Const)
    bv = max(b.val, eps(Float64))
    return (dret.val * (-sign(x.val - μ.val) / bv), nothing, nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# logistic_lpdf(x, μ, s)  — x Active, μ,s Const
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(logistic_lpdf)},
        ::Type{<:Active}, x::Active, μ::Const, s::Const)
    sv = max(s.val, eps(Float64))
    z = (x.val - μ.val) / sv
    lp = -log(sv) - z - 2*log1p(exp(-z))
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(logistic_lpdf)},
        dret::Active, tape, x::Active, μ::Const, s::Const)
    sv = max(s.val, eps(Float64))
    z = (x.val - μ.val) / sv
    sigmoid = 1.0 / (1.0 + exp(-z))
    return (dret.val * ((1.0 - 2.0*sigmoid) / sv), nothing, nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# lognormal_lpdf(x, μ, σ)  — x Active, μ,σ Const
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(lognormal_lpdf)},
        ::Type{<:Active}, x::Active, μ::Const, σ::Const)
    xv = max(x.val, eps(Float64)); σv = max(σ.val, eps(Float64))
    log_x = log(xv); z = (log_x - μ.val) / σv
    lp = -log_x - log(σv) - 0.5*log(2π) - 0.5*z*z
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(lognormal_lpdf)},
        dret::Active, tape, x::Active, μ::Const, σ::Const)
    xv = max(x.val, eps(Float64)); σv = max(σ.val, eps(Float64))
    z = (log(xv) - μ.val) / σv
    return (dret.val * (-1.0/xv - z/(σv*xv)), nothing, nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# gamma_lpdf(x, α, β)  — x Active, α,β Const
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(gamma_lpdf)},
        ::Type{<:Active}, x::Active, α::Const, β::Const)
    xv = max(x.val, eps(Float64)); αv = max(α.val, eps(Float64)); βv = max(β.val, eps(Float64))
    lp = αv * log(βv) - _loggamma(αv) + (αv - 1) * log(xv) - βv * xv
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(gamma_lpdf)},
        dret::Active, tape, x::Active, α::Const, β::Const)
    xv = max(x.val, eps(Float64)); αv = max(α.val, eps(Float64)); βv = max(β.val, eps(Float64))
    return (dret.val * ((αv - 1.0)/xv - βv), nothing, nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# student_t_lpdf(x, ν, μ, σ)  — x Active, rest Const
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(student_t_lpdf)},
        ::Type{<:Active}, x::Active, ν::Const, μ::Const, σ::Const)
    σv = max(σ.val, eps(Float64)); νv = max(ν.val, eps(Float64))
    z = (x.val - μ.val) / σv
    lp = _loggamma(0.5*(νv+1)) - _loggamma(0.5*νv) - 0.5*log(νv*π) -
         log(σv) - 0.5*(νv+1)*log(1 + z^2/νv)
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(student_t_lpdf)},
        dret::Active, tape, x::Active, ν::Const, μ::Const, σ::Const)
    σv = max(σ.val, eps(Float64)); νv = max(ν.val, eps(Float64))
    z = (x.val - μ.val) / σv
    dx = -(νv + 1.0) * z / (νv * σv * (1.0 + z^2/νv))
    return (dret.val * dx, nothing, nothing, nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# beta_lpdf(x, α, β)  — x Active, α,β Const
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(beta_lpdf)},
        ::Type{<:Active}, x::Active, α::Const, β::Const)
    xc = clamp(x.val, eps(Float64), 1.0 - eps(Float64))
    αv = max(α.val, eps(Float64)); βv = max(β.val, eps(Float64))
    lp = (x.val < 0 || x.val > 1) ? -Inf :
         (αv-1)*log(xc) + (βv-1)*log(1-xc) - _logbeta(αv, βv)
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(beta_lpdf)},
        dret::Active, tape, x::Active, α::Const, β::Const)
    xv = clamp(x.val, eps(Float64), 1.0 - eps(Float64))
    αv = max(α.val, eps(Float64)); βv = max(β.val, eps(Float64))
    dx = (αv - 1.0)/xv - (βv - 1.0)/(1.0 - xv)
    return (dret.val * dx, nothing, nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# poisson_lpdf(x, λ)  — x Const, λ Active
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(poisson_lpdf)},
        ::Type{<:Active}, x::Const, λ::Active)
    xv = Float64(x.val); λv = max(λ.val, eps(Float64))
    lp = xv * log(λv) - λv - _loggamma(xv + 1)
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(poisson_lpdf)},
        dret::Active, tape, x::Const, λ::Active)
    xv = Float64(x.val); λv = max(λ.val, eps(Float64))
    return (nothing, dret.val * (xv / λv - 1.0))
end

# ═══════════════════════════════════════════════════════════════════════════════
# binomial_lpdf(x, n, p)  — x,n Const, p Active
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(binomial_lpdf)},
        ::Type{<:Active}, x::Const, n::Const, p::Active)
    xv = Float64(x.val); nv = Float64(n.val)
    pv = clamp(p.val, eps(Float64), 1.0 - eps(Float64))
    log_n_choose_x = -log(nv + 1) - _logbeta(nv - xv + 1, xv + 1)
    lp = log_n_choose_x + xv * log(pv) + (nv - xv) * log(1 - pv)
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(binomial_lpdf)},
        dret::Active, tape, x::Const, n::Const, p::Active)
    xv = Float64(x.val); nv = Float64(n.val)
    pv = clamp(p.val, eps(Float64), 1.0 - eps(Float64))
    dp = xv / pv - (nv - xv) / (1.0 - pv)
    return (nothing, nothing, dret.val * dp)
end

# ═══════════════════════════════════════════════════════════════════════════════
# binomial_logit_lpdf(y, n, α)  — y,n Const, α Active
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(binomial_logit_lpdf)},
        ::Type{<:Active}, y::Const, n::Const, α::Active)
    yv = Float64(y.val); nv = Float64(n.val); αv = α.val
    log_n_choose_y = -log(nv + 1) - _logbeta(nv - yv + 1, yv + 1)
    lp = log_n_choose_y + yv * αv - nv * (log1p(exp(-abs(αv))) + max(0.0, αv))
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(binomial_logit_lpdf)},
        dret::Active, tape, y::Const, n::Const, α::Active)
    sigmoid = 1.0 / (1.0 + exp(-α.val))
    return (nothing, nothing, dret.val * (Float64(y.val) - Float64(n.val) * sigmoid))
end

# ═══════════════════════════════════════════════════════════════════════════════
# weibull_lpdf(x, α, σ)  — x Active, α,σ Const
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(weibull_lpdf)},
        ::Type{<:Active}, x::Active, α::Const, σ::Const)
    xv = max(x.val, eps(Float64)); αv = max(α.val, eps(Float64)); σv = max(σ.val, eps(Float64))
    lp = log(αv) - log(σv) + (αv - 1) * (log(xv) - log(σv)) - (xv / σv)^αv
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(weibull_lpdf)},
        dret::Active, tape, x::Active, α::Const, σ::Const)
    xv = max(x.val, eps(Float64)); αv = max(α.val, eps(Float64)); σv = max(σ.val, eps(Float64))
    dx = (αv - 1.0) / xv - αv * (xv / σv)^(αv - 1.0) / σv
    return (dret.val * dx, nothing, nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# weibull_lccdf(x, α, σ)  — x Active, α,σ Const
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(weibull_lccdf)},
        ::Type{<:Active}, x::Active, α::Const, σ::Const)
    xv = max(x.val, 0.0); αv = max(α.val, eps(Float64)); σv = max(σ.val, eps(Float64))
    lp = -(xv / σv)^αv
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(weibull_lccdf)},
        dret::Active, tape, x::Active, α::Const, σ::Const)
    xv = max(x.val, eps(Float64)); αv = max(α.val, eps(Float64)); σv = max(σ.val, eps(Float64))
    dx = -αv * (xv / σv)^(αv - 1.0) / σv
    return (dret.val * dx, nothing, nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# uniform_lpdf(x, lo, hi)  — gradient is 0
# ═══════════════════════════════════════════════════════════════════════════════

function EnzymeRules.augmented_primal(config, ::Const{typeof(uniform_lpdf)},
        ::Type{<:Active}, x::Active, lo::Const, hi::Const)
    lp = -log(hi.val - lo.val)
    return EnzymeRules.AugmentedReturn(lp, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(uniform_lpdf)},
        dret::Active, tape, x::Active, lo::Const, hi::Const)
    return (0.0, nothing, nothing)
end
