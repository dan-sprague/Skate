## LEAPFROG INTEGRATOR

using Enzyme
using Skater
using Random
data = randn(10)
dim = 2
transforms = (IdentityTransformation(), LogTransformation())
ℓ = build_log_joint(data, normal_lpdf, transforms)

model = ModelLogDensity(dim, ℓ)
q = zeros(Float64,dim)
p = randn(Float64,dim)
grad = zeros(Float64,dim)
state = PhaseSpacePoint(q, p, grad)

∇logp!(state.grad, model, state.q)
allocs = @allocated ∇logp!(state.grad, model, state.q)
allocs == 0


### Q REPRESENTS THE POSITION IN PHASE SPACE OF THE PARAMETERS, P REPRESENTS THE MOMENTUM OF THE PARAMETERS. IN HMC, SAMPLE MOMENTUM FROM A GAUSSIAN DISTRIBUTION, AND THEN SIMULATE THE DYNAMICS OF THE SYSTEM TO PROPOSE NEW SAMPLES.

function leapfrog!(q, p, g, model, ∇logp!, ϵ = 0.1)

    ∇logp!(g, model, q)
    p .+= (ϵ / 2) .* g

    q .+= ϵ .* p

    ∇logp!(g, model, q)
    p .+= (ϵ / 2) .* g

    return nothing

end

struct PhaseSpacePoint
    q::Vector{Float64}
    p::Vector{Float64}
end
struct HMCState
    curr::PhaseSpacePoint
    proposal::PhaseSpacePoint
    grad::Vector{Float64}
end

function hamiltonian(q, p, model)
    return -log_prob(model, q) + 0.5 * sum(abs2, p)
end

## note -- ∇logp! handles zeroing of gradient buffer
function sample!(samples, model, num_samples; ϵ = 0.1, L = 10)
    HMC = HMCState(
        PhaseSpacePoint(zeros(Float64, model.dim), zeros(Float64, model.dim)),
        PhaseSpacePoint(zeros(Float64, model.dim), zeros(Float64, model.dim)),
        zeros(Float64, model.dim)
    )

    @inbounds for i in 1:num_samples
        randn!(HMC.curr.p)
        HMC.proposal.q .= HMC.curr.q
        HMC.proposal.p .= HMC.curr.p
        H_current = hamiltonian(HMC.curr.q, HMC.curr.p, model)
        for _ in 1:L
            leapfrog!(HMC.proposal.q, HMC.proposal.p, HMC.grad, model, ∇logp!, ϵ)
        end
        H_proposal = hamiltonian(HMC.proposal.q, HMC.proposal.p, model)
        α = min(1, exp(H_current - H_proposal))
        if rand() < α
            HMC.curr.q .= HMC.proposal.q
            HMC.curr.p .= HMC.proposal.p
            H_current = H_proposal
        end
        samples[:, i] .= HMC.curr.q
    end
    return nothing 
end

samples = zeros(Float64, model.dim, 1000)
sample!(samples, model, 1000)
