"""
    ModelLogDensity

Compiled model holding the log-density function, parameter dimension,
and a constraining function. Created by `make(data)`.

Fields:
- `dim::Int` — Number of unconstrained parameters.
- `ℓ` — Log-density closure `q::Vector{Float64} → Float64`.
- `constrain` — Maps unconstrained vector to named parameter tuple.
"""
struct ModelLogDensity{F, G}
    dim::Int
    ℓ::F
    constrain::G
end

"""
    log_prob(model, q) → Float64

Evaluate the log-density of `model` at unconstrained parameter vector `q`.
"""
log_prob(m::ModelLogDensity, q) = m.ℓ(q)

## Reverse mode — one forward+backward pass, O(1) in dim
function ∇logp_reverse!(g::Vector{Float64}, ℓ::ModelLogDensity, q::Vector{Float64})
    fill!(g, 0.0)

    result = try
        Enzyme.autodiff(
            Enzyme.ReverseWithPrimal,
            log_prob,
            Enzyme.Active,
            Enzyme.Const(ℓ),
            Enzyme.Duplicated(q, g)
        )
    catch e
        @error "Enzyme autodiff failed" exception = e
        nothing
    end

    if result === nothing
        fill!(g, 0.0)
        return -Inf, false
    end

    lp = result[2]

    if !isfinite(lp) || any(isnan, g) || any(!isfinite, g)
        fill!(g, 0.0)
        return -Inf, false
    end

    return lp, true
end

## Batched forward mode — one widened forward pass, all partials at once
function ∇logp_forward!(g::Vector{Float64}, ℓ::ModelLogDensity, q::Vector{Float64}, seeds)
    fill!(g, 0.0)

    result = try
        Enzyme.autodiff(
            Enzyme.ForwardWithPrimal,
            log_prob,
            Enzyme.Const(ℓ),
            Enzyme.BatchDuplicated(q, seeds)
        )
    catch e
        @error "Enzyme forward autodiff failed" exception = e
        nothing
    end

    if result === nothing
        fill!(g, 0.0)
        return -Inf, false
    end

    lp = result[2]
    derivs = result[1]
    @inbounds for i in eachindex(g)
        g[i] = derivs[i]
    end

    if !isfinite(lp) || any(isnan, g) || any(!isfinite, g)
        fill!(g, 0.0)
        return -Inf, false
    end

    return lp, true
end

