using ForwardDiff
import Base: @kwdef 

abstract type Transformation end

struct IdentityTransformation <: Transformation end
struct LogTransformation <: Transformation end

transform(::IdentityTransformation, x) = x
transform(::LogTransformation, x) = exp(x)

log_abs_det_jacobian(::IdentityTransformation, x) = 0.0
log_abs_det_jacobian(::LogTransformation, x) = x

grad_correction(::IdentityTransformation, x) = 1.0
grad_correction(::LogTransformation, x) = exp(x)

function normal_lpdf(θ, x)
    μ, σ = θ[1], θ[2] 
    return -0.5 * sum((x .- μ).^2) / σ^2 - length(x) * log(σ) - 0.5 * length(x) * log(2π)
end



@kwdef struct TransformedLogDensity{T, F, D}
    data::D
    lpdf::F
    transforms::T 
    buffer::Vector{Float64} 
end

function (model::TransformedLogDensity)(q_unconstrained) 
    logp = 0.0
    q_constrained = [transform(model.transforms[i], q_unconstrained[i]) for i in eachindex(q_unconstrained)]
    for i in eachindex(q_unconstrained)
        logp += log_abs_det_jacobian(model.transforms[i], q_unconstrained[i])
    end

    for x in model.data
        logp += model.lpdf(q_constrained, x)
    end
    
    return logp
end

function ∇logp!(model::TransformedLogDensity, q_unconstrained)
    ForwardDiff.gradient!(model.buffer, model, q_unconstrained)
end


model = TransformedLogDensity(
    data = randn(100),
    lpdf = normal_lpdf,
    transforms = [IdentityTransformation(), LogTransformation()],
    buffer = zeros(2)
)

∇logp!(model, [0.0, 0.0])