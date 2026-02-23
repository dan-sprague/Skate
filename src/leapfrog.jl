## LEAPFROG INTEGRATOR



struct TransformedPhaseSpacePoint
    q::Vector{Float64}
    p::Vector{Float64}
    grad_q::Vector{Float64}
end

normal_lpdf(x,mu,sigma) = -0.5 * log(2π) - log(sigma) - 0.5 * ((x - mu)/sigma)^2

function gradient_log_prob!(pp::PhaseSpacePoint,x)
    for i in eachindex(x)
        pp.grad_q[1] += ( x - pp.q[1]) / pp.q[2]^2
        pp.grad_q[2] += -1 / pp.q[2] + (x - pp.q[1])^2 / pp.q[2]^3
    end
end 




### Q REPRESENTS THE POSITION IN PHASE SPACE OF THE PARAMETERS, P REPRESENTS THE MOMENTUM OF THE PARAMETERS. IN HMC, WE SAMPLE MOMENTUM FROM A GAUSSIAN DISTRIBUTION, AND THEN SIMULATE THE DYNAMICS OF THE SYSTEM TO PROPOSE NEW SAMPLES.

function leapfrog!(state::PhaseSpacePoint,grad,ϵ = 0.1)


end