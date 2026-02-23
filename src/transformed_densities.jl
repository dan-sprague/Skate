using Base: @kwdef 

abstract type Transformation end

struct IdentityTransformation <: Transformation end
struct LogTransformation <: Transformation end

transform(::IdentityTransformation, x) = x
transform(::LogTransformation, x) = exp(x)

log_abs_det_jacobian(::IdentityTransformation, x) = 0.0
log_abs_det_jacobian(::LogTransformation, x) = x

grad_correction(::IdentityTransformation, x) = 1.0
grad_correction(::LogTransformation, x) = exp(x)



@kwdef struct TransformedLogDensity{T<:Tuple, F, D}
    data::D
    lpdf::F
    transforms::T 
    buffer::Vector{Float64} 
end

function (model::TransformedLogDensity)(q_unconstrained::AbstractVector{T}) where {T}
    q_constrained = map(transform, model.transforms, q_unconstrained)
    
    log_jac = zero(T)
    for i in eachindex(model.transforms)
        log_jac += log_abs_det_jacobian(model.transforms[i], q_unconstrained[i])
    end

    log_lik = zero(T)
    for x in model.data
        log_lik += model.lpdf(x, q_constrained...)
    end
    
    return log_jac + log_lik
end

function ∇logp!(model::TransformedLogDensity, q_unconstrained)
    ForwardDiff.gradient!(model.buffer, model, q_unconstrained)
end